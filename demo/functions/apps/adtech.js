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

const cbor = require('cbor')
const util = require('util')
const functions = require('firebase-functions')
const express = require('express')
const cookieParser = require('cookie-parser')
require('dotenv').config({ path: `.env.${process.env.NODE_ENV}` })
const path = require('path')
const { createHash } = require('node:crypto')

const adtech = express()

adtech.use(express.json())
adtech.use(cookieParser())

adtech.set('view engine', 'pug')
adtech.set('views', './views/adtech')
const adtechUrl = process.env.ADTECH_URL
const advertiserUrl = process.env.ADVERTISER_URL
const advertiserNewUrl = process.env.ADVERTISERNEW_URL

adtech.get('/', (req, res) => {
  res.render('index')
})

/* -------------------------------------------------------------------------- */
/*                                     Logging                                */
/* -------------------------------------------------------------------------- */

function log(...args) {
  console.log('\x1b[45m%s\x1b[0m', '[from adtech server] ', ...args)
}

/* -------------------------------------------------------------------------- */
/*                              Key helper functions                          */
/* -------------------------------------------------------------------------- */

// const SCALING_FACTOR_PURCHASE_COUNT = 32768
// const SCALING_FACTOR_PURCHASE_VALUE = 22

function createHashAs64BitHex(input) {
  return createHash('sha256').update(input).digest('hex').substring(0, 16)
}

function generateSourceKeyPiece(input) {
  const hash = createHashAs64BitHex(input)
  const hex = `0x${hash}0000000000000000`
  console.log('generated source key piece: ', hex)
  return hex
}

function generateTriggerKeyPiece(input) {
  const hash = createHashAs64BitHex(input)
  const hex = `0x0000000000000000${hash}`
  console.log('generated trigger key piece: ', hex)
  return hex
}

/* -------------------------------------------------------------------------- */
/*                               Debugging setup                              */
/* -------------------------------------------------------------------------- */

adtech.use(function (req, res, next) {
  console.log(
    'Time:',
    Date.now(),
    ' ',
    req.originalUrl,
    ' Cookies: ',
    req.cookies
  )

  var headers = []
  const legacyMeasurementCookie = req.cookies['__session']
  if (legacyMeasurementCookie === undefined) {
    const cookieValue = Math.floor(Math.random() * 1000000000000000)
    headers.push(`__session=${cookieValue}; SameSite=None; Secure; HttpOnly`)
  }

  // Set the Attribution Reporting debug cookie
  const debugCookie = req.cookies['ar_debug']
  if (debugCookie === undefined) {
    headers.push('ar_debug=1; SameSite=None; Secure; HttpOnly')
  }

  if (headers.length > 0) {
    res.set('Set-Cookie', headers)
  }

  next()
})

/* -------------------------------------------------------------------------- */
/*                                 Ad serving                                 */
/* -------------------------------------------------------------------------- */

adtech.get('/ad-click', (req, res) => {
  res.render('ad-click')
})

adtech.get('/ad-click-new', (req, res) => {
  res.render('ad-click-new')
})

adtech.get('/ad-script-click-element', (req, res) => {
  res.set('Content-Type', 'text/javascript')
  const adClickUrl = `${process.env.ADTECH_URL}/ad-click`
  const iframe = `<iframe src='${adClickUrl}' allow='attribution-reporting' width=190 height=190 scrolling=no frameborder=1 padding=0></iframe>`
  res.send(`document.write("${iframe}");`)
})

adtech.get('/ad-script-click-element-new', (req, res) => {
  res.set('Content-Type', 'text/javascript')
  const adClickUrl = `${process.env.ADTECH_URL}/ad-click-new`
  const iframe = `<iframe src='${adClickUrl}' allow='attribution-reporting' width=190 height=190 scrolling=no frameborder=1 padding=0></iframe>`
  res.send(`document.write("${iframe}");`)
})


/* -------------------------------------------------------------------------- */
/*                  Source registration (ad click or view)                    */
/* -------------------------------------------------------------------------- */
adtech.get('/register-source-href', (req, res) => {
    const attributionDestination = process.env.ADVERTISER_URL
    // For demo purposes, sourceEventId is a random ID. In a real system, this ID would be tied to a unique serving-time identifier mapped to any information an adtech provider may need
    const sourceEventId = Math.floor(Math.random() * 1000000000000000)
    const legacyMeasurementCookie = req.cookies['__session']
    var epoch = 1
    if(req.query['epoch'] != undefined)
    {
      epoch = req.query['epoch']
    }

  const headerConfig = {
    source_event_id: `${sourceEventId}`,
    destination: attributionDestination,
    // Optional: expiry of 7 days (default is 30)
    expiry: '604800',
    // debug_key as legacyMeasurementCookie is a simple approach for demo purposes. In a real system, you may make debug_key a unique ID, and map it to additional source-time information that you deem useful for debugging or performance comparison.
    debug_key: legacyMeasurementCookie,
    epoch: epoch,
    filter_data: {
      campaignId: ['444']
    },
    aggregation_keys: {
      // these source key pieces get binary OR'd with the trigger key piece
      // to create the full histogram bin key
      purchaseCount: generateSourceKeyPiece('COUNT, CampaignID=444'),
      purchaseValue: generateSourceKeyPiece('VALUE, CampaignID=444')
    },
    // optional, but leaving as a comment for future use
    // aggregatable_report_window: "86400" // optional duration in seconds after the source registration during which aggregatable reports can be created for this source.
    debug_reporting: true
  }

  // Send a response with the header Attribution-Reporting-Register-Source in order to instruct the browser to register a source event
  res.set('Attribution-Reporting-Register-Source', JSON.stringify(headerConfig))
  log('REGISTERING SOURCE \n', headerConfig)

  res.redirect(advertiserUrl)

})

adtech.get('/register-source-href-new', (req, res) => {
  const attributionDestination = process.env.ADVERTISERNEW_URL
  // For demo purposes, sourceEventId is a random ID. In a real system, this ID would be tied to a unique serving-time identifier mapped to any information an adtech provider may need
  const sourceEventId = Math.floor(Math.random() * 1000000000000000)
  const legacyMeasurementCookie = req.cookies['__session']
  var epoch = 1
  if(req.query['epoch'] != undefined)
  {
    epoch = req.query['epoch']
  }

const headerConfig = {
  source_event_id: `${sourceEventId}`,
  destination: attributionDestination,
  // Optional: expiry of 7 days (default is 30)
  expiry: '604800',
  // debug_key as legacyMeasurementCookie is a simple approach for demo purposes. In a real system, you may make debug_key a unique ID, and map it to additional source-time information that you deem useful for debugging or performance comparison.
  debug_key: legacyMeasurementCookie,
  epoch: epoch,
  filter_data: {
    campaignId: ['444']
  },
  aggregation_keys: {
    // these source key pieces get binary OR'd with the trigger key piece
    // to create the full histogram bin key
    purchaseCount: generateSourceKeyPiece('COUNT, CampaignID=444'),
    purchaseValue: generateSourceKeyPiece('VALUE, CampaignID=444')
  },
  // optional, but leaving as a comment for future use
  // aggregatable_report_window: "86400" // optional duration in seconds after the source registration during which aggregatable reports can be created for this source.
  debug_reporting: true
}

// Send a response with the header Attribution-Reporting-Register-Source in order to instruct the browser to register a source event
res.set('Attribution-Reporting-Register-Source', JSON.stringify(headerConfig))
log('REGISTERING SOURCE \n', headerConfig)

res.redirect(advertiserNewUrl)

})

/* -------------------------------------------------------------------------- */
/*         Seed source registration for performance benchmarking              */
/* -------------------------------------------------------------------------- */

adtech.get('/seed-source-registration', (req, res) => {
  const attributionDestination = process.env.ADVERTISER_URL
  // For demo purposes, sourceEventId is a random ID. In a real system, this ID would be tied to a unique serving-time identifier mapped to any information an adtech provider may need
  const sourceEventId = Math.floor(Math.random() * 1000000000000000)
  epoch = 2
  if(req.query['epoch'] != undefined)
    epoch = req.query['epoch']
const headerConfig = {
  source_event_id: `${sourceEventId}`,
  destination: attributionDestination,
  // Optional: expiry of 7 days (default is 30)
  expiry: '604800',
  epoch: epoch,
  filter_data: {
    campaignId: ['123']
  },
  aggregation_keys: {
    // these source key pieces get binary OR'd with the trigger key piece
    // to create the full histogram bin key
    purchaseCount: generateSourceKeyPiece('COUNT, CampaignID=123'),
    purchaseValue: generateSourceKeyPiece('VALUE, CampaignID=123')
  },
  // optional, but leaving as a comment for future use
  // aggregatable_report_window: "86400" // optional duration in seconds after the source registration during which aggregatable reports can be created for this source.
  debug_reporting: true
}

// Send a response with the header Attribution-Reporting-Register-Source in order to instruct the browser to register a source event
res.set('Attribution-Reporting-Register-Source', JSON.stringify(headerConfig))
log('REGISTERING SOURCE \n', headerConfig)

res.sendStatus(200)

})

/* -------------------------------------------------------------------------- */
/*                     Attribution trigger (conversion)                       */
/* -------------------------------------------------------------------------- */


adtech.get('/conversion', (req, res) => {
  const productCategory = req.query['product-category']
  // const purchaseValue = req.query['purchase-value']
  var epcoh_end = 2
  var epoch_start = 1
  if(req.query['epoch_end'] != undefined)
  {
    epcoh_end = req.query['epoch_end']
  }

  if(req.query['epoch_start'] != undefined)
  {
    epcoh_end = req.query['epoch_start']
  }

  const filters = {
    // Because conversion_product_type has been set to category_1 in the header Attribution-Reporting-Register-Source, any incoming conversion whose productCategory does not match category_1 will be filtered out i.e. will not generate a report.
    campaignId: ['444']
  }

  const aggregatableTriggerData = [
    // Each dict independently adds pieces to multiple source keys.
    {
      key_piece: generateTriggerKeyPiece(`ProductCategory=${productCategory}`),
      // Apply this key piece to:
      source_keys: ['purchaseCount', 'purchaseValue']
      // source_keys: ['purchaseValue']
    }
  ]

  const aggregatableValues = {
    purchaseCount:  1,
    purchaseValue: 200 //parseInt(purchaseValue)
  }

  const aggregatableCapValues = {
    purchaseCount:  1,
    purchaseValue: 200
  }


  const globalEpsilon = 0.1
  const attributionWindow = {epoch_start: epoch_start, epoch_end: epcoh_end}
  const attributionLogic = "last_touch"
  const partitioningLogic = ""
  
  // Debug report (common to event-level and aggregate)
  console.log('Conversion Cookies Set: ', req.cookies)

  // Optional: set a debug key, and give it the value of the legacy measurement 3P cookie.
  // This is a simple approach for demo purposes. In a real system, you would make this key a unique ID, and you may map it to additional trigger-time information that you deem useful for debugging or performance comparison.
  const legacyMeasurementCookie = req.cookies['__session']

  const headerConfig = {
    filters: filters,
    debug_key: `${legacyMeasurementCookie}`,
    debug_reporting: true
  }

  headerConfig.aggregatable_trigger_data = aggregatableTriggerData
  headerConfig.aggregatable_values = aggregatableValues
  headerConfig.aggregatable_cap_values = aggregatableCapValues
  headerConfig.global_epsilon = globalEpsilon
  headerConfig.attribution_window = attributionWindow
  headerConfig.attribution_logic = attributionLogic
  headerConfig.partitioning_logic = partitioningLogic

  res.set(
    'Attribution-Reporting-Register-Trigger',
    JSON.stringify(headerConfig)
  )

  res.sendStatus(200)
})

/* -------------------------------------------------------------------------- */
/*                                 Reports                                    */
/* -------------------------------------------------------------------------- */

adtech.get('/reports', (req, res) => {
  res.send(JSON.stringify(reports))
})

const decodePlaintextContent = async (encoded) => {
  const hexContent = Buffer.from(encoded, 'base64').toString('hex')
  const result = await cbor.decodeAll(hexContent, {encoding: 'hex'})
  const decoded = result.map((r) => ({
    ...r,
    data: r.data.map((d) => ({
      // bucket is a 128-bit big endian integer, where the first 64 bits (big-endian style, so leading left-to-right)
      // represent the source key piece, and the second 64-bits represent the trigger key piece. But, we're keeping
      // its hex representation now so it is easier to tie with the chrome://attribution-internals
      bucket: `0x${d.bucket.toString('hex')}`, 
      
      // value is a 32-bit big endian integer 
      // the meaning of the value has been scaled up in the /conversion endpoint. see aggregatableValues
      value: d.value.readInt32BE(0),
    })),
  }))
  return decoded
}

// Aggregatable reports
adtech.post(
  '/.well-known/attribution-reporting/report-aggregate-attribution',
  async (req, res) => {
    console.log(
      '\x1b[1;31m%s\x1b[0m',
      `🚀 Adtech has received an aggregatable report from the browser`
    )
    console.log(
      'REGULAR REPORT RECEIVED (aggregate):\n=== \n',
      req.body,
      '\n=== \n'
    )

    res.sendStatus(200)
  }
)

// Primary debug reports for aggregatable
adtech.post(
  '/.well-known/attribution-reporting/debug/report-aggregate-attribution',
  async (req, res) => {
    const aggregation_service_payloads = await Promise.all(req.body.aggregation_service_payloads.map(async (x) => ({
      ...x,
      debug_cleartext_payload: await decodePlaintextContent(x.debug_cleartext_payload),
    })))
    const content = {
      ...req.body,
      aggregation_service_payloads: aggregation_service_payloads,
    }
    console.log(
      '\x1b[1;31m%s\x1b[0m',
      `🚀 Adtech has received a primary debug report for aggregatable from the browser`
    )
    console.log('DEBUG REPORT RECEIVED (aggregate):\n=== ')
    console.log(util.inspect(content, {depth: 10}))
    console.log('=== ')

    res.sendStatus(200)
  }
)

// Verbose debug reports
adtech.post(
  '/.well-known/attribution-reporting/debug/verbose',
  async (req, res) => {
    console.log(
      '\x1b[1;31m%s\x1b[0m',
      `🚀 Adtech has received one or more verbose debug reports from the browser`
    )
    console.log('VERBOSE REPORT(S) RECEIVED:\n=== \n', req.body, '\n=== \n')

    res.sendStatus(200)
  }
)


/* -------------------------------------------------------------------------- */
/*                      Source Registration Seeder Page                       */
/* -------------------------------------------------------------------------- */

adtech.get(
  '/source-registration-seeder',
  async (req, res) => {
    var epoch_start = 1
    var epoch_end = 1
    if(req.query['epoch_start'] != undefined)
      epoch_start = req.query['epoch_start']
    if(req.query['epoch_end'] != undefined)
      epoch_end = req.query['epoch_end']

    res.render('source-registration-seeder', {adtechUrl,epoch_start,epoch_end})
  }
)

exports.adtech = functions.https.onRequest(adtech)
