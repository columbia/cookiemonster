# adsprivate

# Effective On-Device DP Budgeting for Private Attribution Measurement Systems

## Repo Structure

- `chromium`: (submodule) on-device DP budgeting prototyped on Chromium based on Google's Attribution Reporting API.

- `devtools-frontend`: (submodule) frontend changes for extending Attribution Reporting API's Source and Trigger registration parameters.

- `demo`: (submodule) creating servers for adtech, advertiser, publisher using the extended API. (todo: maybe add this directly and not as a submodule)



## Setup `chromium`

#### Build chromium following the instructions here:
https://chromium.googlesource.com/chromium/src/+/main/docs/linux/build_instructions.md

#### Upon making changes to the code:
```bash
autoninja -C out/Default chrome
```

#### Run:
``` bash
out/Default/chrome --disable-gpu --user-data-dir=/mydata/chromium-data --remote-debugging-port=8888 --flag-switches-begin --disable-field-trial-config  --start-maximized --enable-privacy-sandbox-ads-apis --privacy-sandbox-enrollment-overrides=http://arapi-adtech.localhost:8085 --show-overdraw-feedback --flag-switches-end --restore-last-session  http://arapi-publisher.localhost:8087/
```

#### Exporting Display

sudo apt-get install xorg
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
