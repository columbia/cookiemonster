# Attribution Reporting API: demo

## >> [Live demo](https://goo.gle/attribution-reporting-demo)

## Set up and run locally

### Set up (one time only)
1. If it's not yet installed on your machine, install [node.js](https://nodejs.org/en/download/).
   Install using this: https://deb.nodesource.com/ and try >=v20
2. Install the latest Firebase CLI by running the following in your terminal: `npm install -g firebase-tools`. The version known to work with this README.md is 13.3.0.
3. In your terminal, run `git clone https://github.com/columbia/cookiemonster.git && cd cookiemonster/demo`. 
4. In your terminal, run `npm install && cd functions && npm install && cd ..`. This command will install all the required dependencies for you to locally run the `attribution-reporting` demo.

### Run locally
1. Locally start the demo: in your terminal, navigate to `demo` and run `firebase emulators:start --project none`.
    * You should now have multiple servers running: home(:8080), adtech(:8085), advertiser(:8086), publisher(:8087) server.
    * Make sure you see the following output and port mappings in your terminal. If the port mappings differ, see the [#Troubleshooting](#troubleshooting) section.

    ```sh
    ...
    i  hosting[arapi-home]: Serving hosting files from: sites/home
    ✔  hosting[arapi-home]: Local server: http://localhost:8080
    i  hosting[arapi-adtech]: Serving hosting files from: sites/adtech
    ✔  hosting[arapi-adtech]: Local server: http://localhost:8085
    i  hosting[arapi-advertiser]: Serving hosting files from: sites/advertiser
    ✔  hosting[arapi-advertiser]: Local server: http://localhost:8086
    i  hosting[arapi-publisher]: Serving hosting files from: sites/publisher
    ✔  hosting[arapi-publisher]: Local server: http://localhost:8087
    ...
    ```

4. Open [arapi-home.localhost:8080](http://arapi-home.localhost:8080) in Chrome.
5. Follow the instructions in the UI. 🚨 In particular, make sure to follow the **Set up your browser** instructions.
