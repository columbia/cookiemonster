# Demo Scraper
The attribution reporting demo run by selenium.

## Setup
Download `poetry` to manage the dependencies. `pip install -U poetry` if you do not have it. Once installed, run `poetry install` to install the dependencies.

Open up Google Chrome and select a profile you want to use for this demo. Ensure that this profile adheres to the browser requirements and setup requirements outlined at https://arapi-home.web.app.

## Environment Variables
Create a .env file in this directory. Populate it with the environment variables from dotenv file. Fill in the environment variables that make sense for your execution of the trust-safety-demo. The environment variables include:

|NAME|Description|
|---|---|
|USER_PROFILE|The directory where the User Profile to emulate lives. This allows Selenium to load up the browser preferences from the profile.|
|HOME_URL|The arapi-home url|
|PUBLISHER_URL|The arapi-publisher url|
|ADTECH_URL|The arapi-adtech url|
|IMPLICIT_WAIT|The time, in seconds, the Selenium driver will wait for elements to render on the DOM|
|DRIVER_TIMEOUT|The timout for the Selenium driver|
|SLEEP_WHILE_OBSERVING_ATTRIBUTION_INTERNALS|[Optional] If specified, will sleep the demo scraper for the time specified so that you can look at the chrome://attribution-internals the demo created|

## Running
From the command line, run `poetry run python demo_scraper.py`
