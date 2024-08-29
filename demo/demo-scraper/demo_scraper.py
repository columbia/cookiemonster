from dotenv import load_dotenv
import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time


class Config:
    def __init__(self) -> None:
        self.attribution_internals = "chrome://attribution-internals"

    def populate_from_env(self):
        self.home_url = os.getenv("HOME_URL")
        self.publisher_url = os.getenv("PUBLISHER_URL")
        self.adtech_url = os.getenv("ADTECH_URL")
        self.user_profile = os.getenv("USER_PROFILE")
        self.implicit_wait = int(os.getenv("IMPLICIT_WAIT", "10"))
        self.driver_timeout = int(os.getenv("DRIVER_TIMEOUT", "10"))
        self.sleep_while_observing_attribution_internals = int(
            os.getenv("SLEEP_WHILE_OBSERVING_ATTRIBUTION_INTERNALS", "0")
        )


class TrustSafetyDemo:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.window_handles = set()

    def _switch_tabs(self, driver, wait: WebDriverWait) -> None:
        wait.until(EC.number_of_windows_to_be(len(self.window_handles) + 1))

        for window_handle in driver.window_handles:
            if window_handle not in self.window_handles:
                driver.switch_to.window(window_handle)
                self.window_handles.add(window_handle)
                break

    def run(self):

        options = webdriver.ChromeOptions()
        options.add_argument(f"--user-data-dir={self.config.user_profile}")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        with webdriver.Chrome(options=options) as driver:
            driver.get(self.config.home_url)
            driver.implicitly_wait(self.config.implicit_wait)
            wait = WebDriverWait(driver, self.config.driver_timeout)

            self.window_handles.add(driver.current_window_handle)

            publisher_site_link = driver.find_element(
                by=By.LINK_TEXT, value="publisher site"
            )
            publisher_site_link.click()

            self._switch_tabs(driver, wait)

            wait.until(EC.url_matches(self.config.publisher_url))

            ad_that_registers_clicks = driver.find_element(
                by=By.PARTIAL_LINK_TEXT, value="HTML element"
            )
            ad_that_registers_clicks.click()

            wait.until(EC.url_matches(f"{self.config.publisher_url}/click-element"))

            iframe = driver.find_elements(by=By.TAG_NAME, value="iframe")[0]
            driver.switch_to.frame(iframe)

            click_me_link = driver.find_element(by=By.LINK_TEXT, value="Click me")
            wait.until(EC.visibility_of(click_me_link))
            click_me_link.click()

            self._switch_tabs(driver, wait)

            complete_checkout_link = driver.find_element(
                by=By.PARTIAL_LINK_TEXT, value="Complete checkout**"
            )
            wait.until(EC.visibility_of(complete_checkout_link))
            complete_checkout_link.click()

            h3_headers = driver.find_elements(by=By.TAG_NAME, value="h3")
            for h3 in h3_headers:
                if h3.text == "Thanks for your purchase!":
                    wait.until(EC.visibility_of(h3))

            driver.switch_to.new_window("tab")
            driver.get(self.config.attribution_internals)
            wait.until(EC.url_matches(self.config.attribution_internals))

            if self.config.sleep_while_observing_attribution_internals:
                time.sleep(self.config.sleep_while_observing_attribution_internals)


if __name__ == "__main__":
    load_dotenv()
    config = Config()
    config.populate_from_env()

    trust_safety_demo = TrustSafetyDemo(config)
    trust_safety_demo.run()
