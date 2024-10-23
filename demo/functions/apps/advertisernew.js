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
const session = require('express-session')

const PORT = 8082
const demoHomeUrl = process.env.DEMO_HOME_URL
const publisherUrl = process.env.PUBLISHER_URL
const advertiserUrl = process.env.ADVERTISER_URL
const adtechUrl = process.env.ADTECH_URL
const adtechNewUrl = process.env.ADTECHNEW_URL
const advertiserNewUrl = process.env.ADVERTISERNEW_URL

const advertisernew = express()
advertisernew.set('view engine', 'pug')
advertisernew.set('views', './views/advertisernew')
advertisernew.use(
  session({
    secret: '343ji43j4n3jn4jk3n',
    saveUninitialized: true,
    resave: true
  })
)

advertisernew.use(express.json())

// Middleware run on each request
advertisernew.use((req, res, next) => {
  // Check if session is initialized
  if (!req.session.initialized) {
    // Initialize variables on the session object (persisted across requests made by the same user)
    req.session.initialized = true
    req.session.prio = false
    req.session.dedup = false
  }
  next()
})

advertisernew.get('/', (req, res) => {
  res.render('home', { demoHomeUrl, publisherUrl, advertiserNewUrl, adtechUrl })
})

advertisernew.post('/new-purchase', (req, res) => {
  req.session.purchaseId = Math.floor(Math.random() * 100000)
  res.redirect('checkout')
})

advertisernew.get('/checkout', (req, res) => {
  if (!req.session.purchaseId) {
    req.session.purchaseId = Math.floor(Math.random() * 100000)
  }
  const { prio, dedup, purchaseId } = req.session

  const searchParams = new URLSearchParams({
    'conversion-type': 'checkout-completed',
    'product-category': 'category_1',
    'purchase-value': 200,
    'prio-checkout': prio,
    dedup: dedup,
    'purchase-id': purchaseId
  })

  const adtechRequestUrl = `${
    process.env.ADTECH_URL
  }/conversion?${searchParams.toString()}`

  const adtechNewRequestUrl = `${
    process.env.ADTECHNEW_URL
  }/conversion?${searchParams.toString()}`

  res.render('checkout', {
    adtechRequestUrl,
    adtechNewRequestUrl,
    purchaseId,
    demoHomeUrl,
    publisherUrl,
    advertiserNewUrl,
    adtechUrl,
    adtechNewUrl
  })
})

exports.advertisernew = functions.https.onRequest(advertisernew)