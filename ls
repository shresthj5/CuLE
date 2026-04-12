[33mcommit 338728a61884e4bee6847b8d1786f1d97ebcd907[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmaster[m[33m)[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Sat Apr 11 19:49:54 2026 -0400

    more fixes for Atari 5 + 2 others

[33mcommit f29921bb8a19ac278ba26acea5442a889eac2b66[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Sat Apr 11 15:35:57 2026 -0400

    all 9 pass 16 step deterministic step

[33mcommit 5f90bff1490cd107055b535c9ce4a988264c8848[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Sat Apr 11 13:07:53 2026 -0400

    8/9 fixed

[33mcommit d42d8e8b1edeb8259715bd71892df936309a5049[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Fri Apr 10 16:46:08 2026 -0400

    more fixes regarding black box pixel to cpu comparison

[33mcommit 982c90f2603014bb627543852397ae35ecb405a5[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Fri Apr 10 15:01:41 2026 -0400

    cpu cuda for 13 games fix

[33mcommit 28013f4ac08f619fb87467e112a509c355c050b7[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Thu Apr 9 20:59:24 2026 -0400

    kernel fixes

[33mcommit ebb4f161b5d9d7027655c692889e119fd4e5b2b5[m[33m ([m[1;31morigin/master[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Thu Apr 9 19:14:17 2026 -0400

    Save modernization changes

[33mcommit 73f9978fb71e9c0fb27cbaeb111964a48765d6d4[m
Author: shresthj5 <142252398+shresthj5@users.noreply.github.com>
Date:   Wed Apr 8 19:28:44 2026 -0400

    Spec Addition

[33mcommit dd0382b99ded6be23cd3c3e79e37938e7c873de0[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Wed Nov 16 22:37:15 2022 -0500

    Updates to Dockerfile for new software versions

[33mcommit d66712ec5cba2f80c9bbc7fd1d108bbe67b886ca[m
Merge: 513825f ed940cf
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Wed Jan 19 11:56:26 2022 -0500

    Merge branch 'master' of github.com:NVlabs/cule

[33mcommit 513825fb3a747054e90dc505bed961036be35b19[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Wed Jan 19 11:56:20 2022 -0500

    Fixed default build for systems with pytorch but no available GPUs, updated method to find installed ROMs

[33mcommit ed940cf90a582bcb20b27cdd223f9cefa258e66e[m
Author: iuri frosio <ifrosio@nvidia.com>
Date:   Fri Jan 7 10:16:38 2022 +0100

    Update README.md

[33mcommit bdddab9d86b5f4914dc6f6ebb4ffb28f856a18d1[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Tue Sep 21 11:05:13 2021 -0400

    Removed travis ci yaml file

[33mcommit 27bd17d9d3ac140d80289c971d6d01b919301ab7[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 22:38:34 2020 -0500

    Added atari save/load state

[33mcommit df4b680a132cb1bb55e72760470dc3b4282a3d7f[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 22:38:02 2020 -0500

    Removed c++14 std

[33mcommit 99416dfdaa94afea9bf270ecca9db82d49c9cd9b[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 14:42:21 2020 -0500

    Initial support for two-player games

[33mcommit 054159f7012e441ea5a9a4b22283d202d051510a[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 13:35:22 2020 -0500

    Removed bits of amp logic from PPO and VTrace examples

[33mcommit 7eff060fa727bf1acf2321c61be2933de5a52542[m
Merge: a868d87 40adc30
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 13:16:32 2020 -0500

    Merge branch 'master' of github.com:NVlabs/cule into develop

[33mcommit a868d876f1d276d7012854bde7037140961f7948[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 13:15:28 2020 -0500

    Replaced amp DDP with native PyTorch

[33mcommit 5d3b1098964bd8344e62cb3e96b8269beaad322e[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 13:14:42 2020 -0500

    Moved dqn agent to native pytorch DDP

[33mcommit 1de1803bc61d278f808c64e9cd88b09449054f8c[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 13:13:46 2020 -0500

    Fixed replay buffer index error

[33mcommit 324251c3e651393e105014b2332440b5a6754421[m
Author: Steven Dalton <sdalton@nvidia.com>
Date:   Thu Dec 3 13:12:24 2020 -0500

    Ignore minor architectures during build
