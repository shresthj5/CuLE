#include <cule/cule.hpp>
#include <cule/cuda.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <torchcule/atari_env.hpp>
#include <torchcule/atari_state.hpp>
#include <torchcule/atari_state.cpp>

#include <algorithm>
#include <sstream>
#include <vector>

using cule_policy = cule::cuda::parallel_execution_policy;

namespace
{
constexpr size_t FRAME_BUFFER_BYTES = 300 * cule::atari::SCREEN_WIDTH;

template<typename T>
class device_buffer
{
public:
    explicit device_buffer(const size_t count)
        : ptr_(nullptr)
    {
        if(count != 0)
        {
            CULE_ERRCHK(cudaMalloc(&ptr_, sizeof(T) * count));
        }
    }

    ~device_buffer()
    {
        if(ptr_ != nullptr)
        {
            CULE_ERRCHK(cudaFree(ptr_));
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* data()
    {
        return static_cast<T*>(ptr_);
    }

    const T* data() const
    {
        return static_cast<const T*>(ptr_);
    }

private:
    void* ptr_;
};

__global__ void
apply_sticky_actions_kernel(const size_t num_envs,
                            const uint8_t* sticky_mask,
                            const uint8_t* requested_actions,
                            const uint8_t* previous_actions,
                            uint8_t* output_actions)
{
    const size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < num_envs)
    {
        output_actions[index] =
            sticky_mask[index] != 0 ? previous_actions[index] : requested_actions[index];
    }
}
} // namespace

AtariEnv::
AtariEnv(const cule::atari::rom& cart,
         const size_t num_envs,
         const size_t noop_reset_steps)
    : super_t(cart, num_envs, noop_reset_steps),
      cule_par(new agency::parallel_execution_policy()),
      num_channels(0),
      rescale(false),
      use_cuda(false),
      gpu_id(-1),
      sticky_actions_enabled(false),
      sticky_threshold(0),
      sticky_mask_host(nullptr),
      sticky_mask_device(nullptr),
      sticky_mask_capacity(0)
{
}

AtariEnv::
~AtariEnv()
{
    release_sticky_mask_buffers();

    if(use_cuda)
    {
        delete &get_policy<cule_policy>();
    }
    else
    {
        delete &get_policy<agency::parallel_execution_policy>();
    }
}

void
AtariEnv::
ensure_sticky_mask_buffers()
{
    if(!use_cuda || (size() == 0))
    {
        return;
    }

    if(sticky_mask_capacity >= size())
    {
        return;
    }

    release_sticky_mask_buffers();

    if(gpu_id != -1)
    {
        CULE_ERRCHK(cudaSetDevice(gpu_id));
    }

    CULE_ERRCHK(cudaMallocHost(reinterpret_cast<void**>(&sticky_mask_host),
                               sizeof(uint8_t) * size()));
    CULE_ERRCHK(cudaMalloc(reinterpret_cast<void**>(&sticky_mask_device),
                           sizeof(uint8_t) * size()));
    sticky_mask_capacity = size();
}

void
AtariEnv::
release_sticky_mask_buffers()
{
    if(gpu_id != -1)
    {
        CULE_ERRCHK(cudaSetDevice(gpu_id));
    }

    if(sticky_mask_device != nullptr)
    {
        CULE_ERRCHK(cudaFree(sticky_mask_device));
        sticky_mask_device = nullptr;
    }

    if(sticky_mask_host != nullptr)
    {
        CULE_ERRCHK(cudaFreeHost(sticky_mask_host));
        sticky_mask_host = nullptr;
    }

    sticky_mask_capacity = 0;
}

void
AtariEnv::
seed_sticky_actions(const uint32_t* seedBuffer,
                    const bool enabled,
                    const uint64_t sticky_threshold)
{
    sticky_actions_enabled = enabled;
    this->sticky_threshold = sticky_threshold;
    sticky_random_states.clear();

    if(!sticky_actions_enabled || (size() == 0))
    {
        return;
    }

    sticky_random_states.resize(size());
    std::vector<uint32_t> host_seeds(size(), 0);

    if(use_cuda)
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        CULE_ERRCHK(cudaMemcpyAsync(host_seeds.data(),
                                    seedBuffer,
                                    sizeof(uint32_t) * size(),
                                    cudaMemcpyDeviceToHost,
                                    stream));
        CULE_ERRCHK(cudaStreamSynchronize(stream));
        ensure_sticky_mask_buffers();
    }
    else
    {
        std::copy(seedBuffer, seedBuffer + size(), host_seeds.begin());
    }

    for(size_t index = 0; index < host_seeds.size(); ++index)
    {
        sticky_random_states[index].seed(host_seeds[index]);
    }
}

void
AtariEnv::
apply_exact_sticky_actions(const uint8_t* requestedActions,
                           const uint8_t* previousActions,
                           uint8_t* outputActions)
{
    if(!sticky_actions_enabled || (size() == 0))
    {
        return;
    }

    if(use_cuda)
    {
        ensure_sticky_mask_buffers();

        for(size_t index = 0; index < size(); ++index)
        {
            sticky_mask_host[index] =
                static_cast<uint64_t>(sticky_random_states[index]()) < sticky_threshold ? 1U : 0U;
        }

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        CULE_ERRCHK(cudaMemcpyAsync(sticky_mask_device,
                                    sticky_mask_host,
                                    sizeof(uint8_t) * size(),
                                    cudaMemcpyHostToDevice,
                                    stream));

        constexpr size_t BLOCK_SIZE = 256;
        const size_t num_blocks = (size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_sticky_actions_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            size(),
            sticky_mask_device,
            requestedActions,
            previousActions,
            outputActions);
        CULE_ERRCHK(cudaPeekAtLastError());
    }
    else
    {
        for(size_t index = 0; index < size(); ++index)
        {
            outputActions[index] =
                static_cast<uint64_t>(sticky_random_states[index]()) < sticky_threshold
                    ? previousActions[index]
                    : requestedActions[index];
        }
    }
}

void
AtariEnv::
reset(uint32_t* seedBuffer,
      uint32_t* aleSeedBuffer)
{
    if(use_cuda)
    {
        super_t::reset(get_policy<cule_policy>(), seedBuffer, aleSeedBuffer);
    }
    else
    {
        super_t::reset(get_policy<agency::parallel_execution_policy>(), seedBuffer, aleSeedBuffer);
    }

    if(sticky_actions_enabled && (reset_sticky_rng_draws != 0))
    {
        for(auto& rng : sticky_random_states)
        {
            for(uint64_t draw = 0; draw < reset_sticky_rng_draws; ++draw)
            {
                rng();
            }
        }
    }
}

void
AtariEnv::
configure_reset_semantics(const bool ale_reset_semantics,
                          const uint32_t frame_skip,
                          const float repeat_action_probability)
{
    super_t::configure_reset_semantics(ale_reset_semantics,
                                       frame_skip,
                                       repeat_action_probability);
}

void
AtariEnv::
reset_states()
{
    if(use_cuda)
    {
        super_t::reset_states(get_policy<cule_policy>());
    }
    else
    {
        super_t::reset_states(get_policy<agency::parallel_execution_policy>());
    }
}

void
AtariEnv::
get_states(const size_t num_states,
           const int32_t* indices_ptr,
           AtariState* states_ptr)
{
    agency::vector<cule::atari::state> output_states(num_states);
    agency::vector<cule::atari::frame_state> output_frame_states(num_states);
    agency::vector<uint8_t> output_states_ram(256 * num_states);

    if(use_cuda)
    {
        auto& policy = get_policy<cule_policy>();
        device_buffer<cule::atari::state> output_states_gpu(num_states);
        device_buffer<cule::atari::frame_state> output_frame_states_gpu(num_states);
        device_buffer<uint8_t> output_states_ram_gpu(256 * num_states);
        device_buffer<int32_t> indices_gpu(num_states);

        CULE_ERRCHK(cudaMemcpyAsync(indices_gpu.data(),
                                    indices_ptr,
                                    sizeof(int32_t) * num_states,
                                    cudaMemcpyHostToDevice,
                                    policy.getStream()));

        super_t::get_states(policy, num_states, indices_gpu.data(), output_states_gpu.data(), output_frame_states_gpu.data(), output_states_ram_gpu.data());
        policy.sync();

        CULE_ERRCHK(cudaMemcpy(output_states.data(),
                               output_states_gpu.data(),
                               sizeof(cule::atari::state) * num_states,
                               cudaMemcpyDeviceToHost));
        CULE_ERRCHK(cudaMemcpy(output_frame_states.data(),
                               output_frame_states_gpu.data(),
                               sizeof(cule::atari::frame_state) * num_states,
                               cudaMemcpyDeviceToHost));
        CULE_ERRCHK(cudaMemcpy(output_states_ram.data(),
                               output_states_ram_gpu.data(),
                               sizeof(uint8_t) * 256 * num_states,
                               cudaMemcpyDeviceToHost));
    }
    else
    {
        super_t::get_states(get_policy<agency::parallel_execution_policy>(), num_states, indices_ptr, output_states.data(), output_frame_states.data(), nullptr);
    }

    // Keep state decoding sequential. This bridge touches non-trivial host-side
    // Python-facing structures and is not on the emulator hot path.
    agency::bulk_invoke(agency::seq(num_states),
                        decode_states_functor{},
                        use_cuda,
                        this->cart,
                        states_ptr,
                        output_states.data(),
                        output_frame_states.data(),
                        output_states_ram.data());

    for(size_t state_index = 0; state_index < num_states; ++state_index)
    {
        const int32_t env_index = indices_ptr[state_index];
        AtariState& atari_state = states_ptr[state_index];

        if(use_cuda)
        {
            CULE_ERRCHK(cudaMemcpy(atari_state.frame_buffer.data(),
                                   this->frame_ptr + (env_index * FRAME_BUFFER_BYTES),
                                   FRAME_BUFFER_BYTES,
                                   cudaMemcpyDeviceToHost));
            CULE_ERRCHK(cudaMemcpy(atari_state.previous_frame_buffer.data(),
                                   this->previous_frame_ptr + (env_index * FRAME_BUFFER_BYTES),
                                   FRAME_BUFFER_BYTES,
                                   cudaMemcpyDeviceToHost));
        }
        else
        {
            std::copy(this->frame_ptr + (env_index * FRAME_BUFFER_BYTES),
                      this->frame_ptr + ((env_index + 1) * FRAME_BUFFER_BYTES),
                      atari_state.frame_buffer.begin());
            std::copy(this->previous_frame_ptr + (env_index * FRAME_BUFFER_BYTES),
                      this->previous_frame_ptr + ((env_index + 1) * FRAME_BUFFER_BYTES),
                      atari_state.previous_frame_buffer.begin());
        }

        if(sticky_actions_enabled &&
           (static_cast<size_t>(env_index) < sticky_random_states.size()))
        {
            std::ostringstream stream;
            stream << sticky_random_states[env_index];
            atari_state.sticky_rng_state = stream.str();
        }
        else
        {
            atari_state.sticky_rng_state.clear();
        }
    }
}

void
AtariEnv::
set_states(const size_t num_states,
           const int32_t* indices_ptr,
           const AtariState* states_ptr)
{
    agency::vector<cule::atari::state> input_states(num_states);
    agency::vector<cule::atari::frame_state> input_frame_states(num_states);
    agency::vector<uint8_t> input_states_ram(256 * num_states);

    agency::bulk_invoke(agency::seq(num_states),
                        encode_states_functor{},
                        this->cart,
                        states_ptr,
                        input_states.data(),
                        input_frame_states.data(),
                        input_states_ram.data());

    if(use_cuda)
    {
        auto& policy = get_policy<cule_policy>();
        device_buffer<cule::atari::state> input_states_gpu(num_states);
        device_buffer<cule::atari::frame_state> input_frame_states_gpu(num_states);
        device_buffer<uint8_t> input_states_ram_gpu(256 * num_states);
        device_buffer<int32_t> indices_gpu(num_states);

        CULE_ERRCHK(cudaMemcpyAsync(input_states_gpu.data(),
                                    input_states.data(),
                                    sizeof(cule::atari::state) * num_states,
                                    cudaMemcpyHostToDevice,
                                    policy.getStream()));
        CULE_ERRCHK(cudaMemcpyAsync(input_frame_states_gpu.data(),
                                    input_frame_states.data(),
                                    sizeof(cule::atari::frame_state) * num_states,
                                    cudaMemcpyHostToDevice,
                                    policy.getStream()));
        CULE_ERRCHK(cudaMemcpyAsync(input_states_ram_gpu.data(),
                                    input_states_ram.data(),
                                    sizeof(uint8_t) * 256 * num_states,
                                    cudaMemcpyHostToDevice,
                                    policy.getStream()));
        CULE_ERRCHK(cudaMemcpyAsync(indices_gpu.data(),
                                    indices_ptr,
                                    sizeof(int32_t) * num_states,
                                    cudaMemcpyHostToDevice,
                                    policy.getStream()));

        super_t::set_states(policy, num_states, indices_gpu.data(), input_states_gpu.data(),
                            input_frame_states_gpu.data(), input_states_ram_gpu.data());
        policy.sync();

        for(size_t state_index = 0; state_index < num_states; ++state_index)
        {
            const int32_t env_index = indices_ptr[state_index];
            const AtariState& atari_state = states_ptr[state_index];
            CULE_ERRCHK(cudaMemcpy(this->frame_ptr + (env_index * FRAME_BUFFER_BYTES),
                                   atari_state.frame_buffer.data(),
                                   FRAME_BUFFER_BYTES,
                                   cudaMemcpyHostToDevice));
            CULE_ERRCHK(cudaMemcpy(this->previous_frame_ptr + (env_index * FRAME_BUFFER_BYTES),
                                   atari_state.previous_frame_buffer.data(),
                                   FRAME_BUFFER_BYTES,
                                   cudaMemcpyHostToDevice));
        }
    }
    else
    {
        super_t::set_states(get_policy<agency::parallel_execution_policy>(), num_states, indices_ptr,
                            input_states.data(), input_frame_states.data(), input_states_ram.data());

        for(size_t state_index = 0; state_index < num_states; ++state_index)
        {
            const int32_t env_index = indices_ptr[state_index];
            const AtariState& atari_state = states_ptr[state_index];
            this->frame_states_ptr[env_index].frameBufferIndex = atari_state.frame_buffer_index;
            std::copy(atari_state.frame_buffer.begin(),
                      atari_state.frame_buffer.end(),
                      this->frame_ptr + (env_index * FRAME_BUFFER_BYTES));
            std::copy(atari_state.previous_frame_buffer.begin(),
                      atari_state.previous_frame_buffer.end(),
                      this->previous_frame_ptr + (env_index * FRAME_BUFFER_BYTES));
        }
    }

    if(sticky_actions_enabled)
    {
        for(size_t state_index = 0; state_index < num_states; ++state_index)
        {
            const int32_t env_index = indices_ptr[state_index];
            const AtariState& atari_state = states_ptr[state_index];
            if(!atari_state.sticky_rng_state.empty() &&
               (static_cast<size_t>(env_index) < sticky_random_states.size()))
            {
                std::istringstream stream(atari_state.sticky_rng_state);
                stream >> sticky_random_states[env_index];
            }
        }
    }
}

void
AtariEnv::
step(const bool fire_reset,
     const cule::atari::Action* playerABuffer,
     const cule::atari::Action* playerBBuffer,
     bool* doneBuffer)
{
    if(use_cuda)
    {
        super_t::step(get_policy<cule_policy>(), fire_reset, playerABuffer, playerBBuffer, doneBuffer);
    }
    else
    {
        super_t::step(get_policy<agency::parallel_execution_policy>(), fire_reset, playerABuffer, playerBBuffer, doneBuffer);
    }
}

void
AtariEnv::
get_data(const bool episodic_life,
         bool* doneBuffer,
         float* rewardsBuffer,
         int32_t* livesBuffer)
{
    if(use_cuda)
    {
        super_t::get_data(get_policy<cule_policy>(), episodic_life, doneBuffer, rewardsBuffer, livesBuffer);
    }
    else
    {
        super_t::get_data(get_policy<agency::parallel_execution_policy>(), episodic_life, doneBuffer, rewardsBuffer, livesBuffer);
    }
}

void
AtariEnv::
generate_frames(const bool rescale,
                const bool last_frame,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    if(use_cuda)
    {
        super_t::generate_frames(get_policy<cule_policy>(), rescale, last_frame, num_channels, imageBuffer);
    }
    else
    {
        super_t::generate_frames(get_policy<agency::parallel_execution_policy>(), rescale, last_frame, num_channels, imageBuffer);
    }
}

void
AtariEnv::
generate_reset_screen_frames(const bool rescale,
                             const size_t num_channels,
                             uint8_t* imageBuffer)
{
    if(use_cuda)
    {
        super_t::generate_reset_screen_frames(get_policy<cule_policy>(), rescale, num_channels, imageBuffer);
    }
    else
    {
        super_t::generate_reset_screen_frames(get_policy<agency::parallel_execution_policy>(), rescale, num_channels, imageBuffer);
    }
}

void
AtariEnv::
generate_random_actions(cule::atari::Action* actionBuffer)
{
    if(use_cuda)
    {
        super_t::generate_random_actions(get_policy<cule_policy>(), actionBuffer);
    }
    else
    {
        super_t::generate_random_actions(get_policy<agency::parallel_execution_policy>(), actionBuffer);
    }
}

void
AtariEnv::
sync_other_stream(cudaStream_t& stream)
{
    if(use_cuda)
    {
        get_policy<cule_policy>().insert_other_stream(stream);
    }
}

void
AtariEnv::
sync_this_stream(cudaStream_t& stream)
{
    if(use_cuda)
    {
        get_policy<cule_policy>().insert_this_stream(stream);
    }
}

void
AtariEnv::
set_cuda(const bool use_cuda, const int32_t gpu_id)
{
    if(this->use_cuda != use_cuda)
    {
        release_sticky_mask_buffers();
        this->use_cuda = use_cuda;

        if(use_cuda)
        {
            this->gpu_id = gpu_id;
            CULE_ERRCHK(cudaSetDevice(gpu_id));
            delete &get_policy<agency::parallel_execution_policy>();
            cule_par = new cule_policy();
        }
        else
        {
            this->gpu_id = -1;
            delete &get_policy<cule_policy>();
            cule_par = new agency::parallel_execution_policy();
        }
    }
}

template<typename ExecutionPolicy>
ExecutionPolicy&
AtariEnv::
get_policy()
{
    assert(cule_par != nullptr);
    if(gpu_id != -1)
    {
        CULE_ERRCHK(cudaSetDevice(gpu_id));
    }

    return *reinterpret_cast<ExecutionPolicy*>(cule_par);
}

size_t
AtariEnv::
state_size()
{
    return sizeof(cule::atari::state);
}

size_t
AtariEnv::
frame_state_size()
{
    return sizeof(cule::atari::frame_state);
}

size_t
AtariEnv::
tia_update_size()
{
    return cule::atari::ENV_UPDATE_SIZE;
}

#include <cule/atari/rom.cpp>
#include <cule/atari/wrapper.cpp>
