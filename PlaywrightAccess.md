# Accessing Playwright in Cursor + WSL

This repo can use the installed Codex `playwright` skill, but in this environment there were a few setup details needed before it worked.

## What was needed

1. Node.js, `npm`, and `npx`
2. The `playwright` skill installed under `~/.codex/skills/playwright`
3. A browser runtime installed for Playwright
4. Loading `nvm` in the shell, because Cursor's Codex session did not automatically inherit it

## 1. Install Node with `nvm`

Run this in your WSL terminal:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

nvm install 24
node -v
npm -v
npx -v
```

Expected result:

- `node -v` prints something like `v24.14.1`
- `npm -v` prints something like `11.11.0`
- `npx -v` should also work

## 2. Load `nvm` in shells that need Playwright

If a fresh shell cannot find `node`, `npm`, or `npx`, run:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
```

This mattered in Cursor/Codex because the agent shell did not automatically pick up the `nvm` environment.

## 3. Confirm the Playwright skill is installed

The skill should exist here:

```bash
ls ~/.codex/skills/playwright
```

The wrapper script is here:

```bash
~/.codex/skills/playwright/scripts/playwright_cli.sh
```

In this install, the wrapper script was not marked executable, so invoking it through `bash` was the reliable option.

## 4. Set the helper variables

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
```

## 5. Check the CLI

```bash
bash "$PWCLI" --help
```

If that works, Playwright is reachable from the terminal.

## 6. Install a browser runtime

These worked without needing a system Chrome install:

```bash
bash "$PWCLI" install-browser firefox
bash "$PWCLI" install-browser webkit
```

`chrome` was not the best choice in this environment because the CLI tried to use a system Chrome path and requested extra privileged setup.

## 7. Open a page

Firefox worked in this environment:

```bash
bash "$PWCLI" open about:blank --browser firefox
```

Then navigate somewhere:

```bash
bash "$PWCLI" goto https://playwright.dev
bash "$PWCLI" eval 'JSON.stringify({title: document.title, href: location.href})' --raw
```

## 8. Useful commands

```bash
bash "$PWCLI" snapshot
bash "$PWCLI" screenshot
bash "$PWCLI" tab-list
bash "$PWCLI" close
```

## 9. Typical working session

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

bash "$PWCLI" open about:blank --browser firefox
bash "$PWCLI" goto https://docs.pytorch.org/docs/stable/cpp_extension.html
bash "$PWCLI" snapshot
bash "$PWCLI" eval 'JSON.stringify({title: document.title, href: location.href})' --raw
bash "$PWCLI" close
```

## Notes for Cursor

- If Cursor or Codex says `npx` is missing, the fix is usually to source `~/.nvm/nvm.sh` again in that shell.
- If `playwright_cli.sh` gives `Permission denied`, run it as `bash "$PWCLI"` instead of executing it directly.
- If a browser fails to launch, try `firefox` first in this environment.
