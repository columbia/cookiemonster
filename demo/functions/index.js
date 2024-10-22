
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

const home = require('./apps/home');
const adtech = require('./apps/adtech');
const advertiser = require('./apps/advertiser');
const publisher = require('./apps/publisher');
const news = require('./apps/news');
const blog = require('./apps/blog');


exports.home = home.home
exports.adtech = adtech.adtech
exports.advertiser = advertiser.advertiser
exports.publisher = publisher.publisher
exports.news = news.news
exports.blog = blog.blog


