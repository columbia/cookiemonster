## Setup `chromium`

Don't initialize the chromium submodule. We keep it here just for quick access to the chromuim repo. Use the following instructions to download/install our chromium instead.

#### Clone depot_tools:
```bash
git clone https://github.com/columbia/depot_tools.git
```

#### Add depot_tools to PATH
```bash
export PATH="/path/to/depot_tools:$PATH"
```

#### Create a directory to download chromium
```bash
mkdir chromium && cd chromium
```

#### Fetch chromium
```bash
fetch --nohooks chromium
```

Press `n` to the following:
`OK to update it to https://chromium.googlesource.com/chromium/tools/depot_tools.git ? [Y/n] n`\
(don't press `enter` after pressing `n`!)

#### Install dependencies
```bash
cd src
./build/install-build-deps.sh
```

#### Run the hooks
```bash
gclient runhooks
```

Press `n` to the following:
`OK to update it to https://chromium.googlesource.com/chromium/tools/depot_tools.git ? [Y/n] n`\
(don't press `enter` after pressing `n`!)

#### Setup the build
```bash
gn gen out/Default
```


#### Pull changes from our devtools-frontend 
Hacky way to pull changes from our devtools-frontend. 
Previous command fetched the google repos instead. TODO: fix that.
```bash
cd third_party/devtools-frontend/src/
git remote add cu https://github.com/columbia/devtools-frontend.git
git fetch cu
git checkout cu/cu-ara
```


#### Build chromium
```bash
cd ../../../../
git checkout origin/logging-filters
autoninja -C out/Default chrome
```

#### Setup Exporting Display (if running Chromium in a remote node)
```bash
sudo apt-get install xorg
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```
Also make sure you SSH using the -X flag and download a tool like XQuartz in your local machine.


#### Run Chromium:
``` bash
out/Default/chrome --disable-gpu --user-data-dir=/mydata/chromium-data --remote-debugging-port=8888 --flag-switches-begin --disable-field-trial-config  --start-maximized --enable-privacy-sandbox-ads-apis --privacy-sandbox-enrollment-overrides=http://adtech.localhost:8085 --show-overdraw-feedback --flag-switches-end --restore-last-session  http://publisher.localhost:8087/
```

