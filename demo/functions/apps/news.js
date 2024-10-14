/**
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const functions = require('firebase-functions')
const express = require('express')
require('dotenv').config({ path: `.env.${process.env.NODE_ENV}` })

const news = express()
news.set('view engine', 'pug')
news.set('views', './views/news')

const demoHomeUrl = process.env.DEMO_HOME_URL
const newsUrl = process.env.NEWS_URL
const advertiserUrl = process.env.ADVERTISER_URL
const adtechUrl = process.env.ADTECH_URL

news.get('/', (req, res) => {
  // TODO: pass some context along with the URL. Like cookie ID or specific article.
  // const source = encodeURIComponent(req.originalUrl) // Retrieve and encode the current page URL
  const adScriptUrl = `${process.env.ADTECH_URL}/ad-script-click-element?source=${newsUrl}`
  res.render('article', {
    adScriptUrl,
    demoHomeUrl,
    newsUrl,
    advertiserUrl,
    adtechUrl
  })
})

exports.news = functions.https.onRequest(news)
