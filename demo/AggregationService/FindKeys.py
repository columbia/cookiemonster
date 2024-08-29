import base64
import cbor2
import re
debugCleartextPayload = "omRkYXRhlKJldmFsdWVEAAAAAWZidWNrZXRQ5/ME/vcoO+0m1RjAlowp3KJldmFsdWVEAAAAyGZidWNrZXRQrkrJMc0P/NAm1RjAlowp3KJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAKJldmFsdWVEAAAAAGZidWNrZXRQAAAAAAAAAAAAAAAAAAAAAGlvcGVyYXRpb25paGlzdG9ncmFt"
cbor_data = base64.b64decode(debugCleartextPayload)
decoded_cbor = cbor2.loads(cbor_data)


buckets = [''.join(f'\\u{byte:04x}' for byte in bucket) for item in decoded_cbor['data'] for bucket in [item['bucket']]]
print(buckets)