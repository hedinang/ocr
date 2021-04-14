from Crypto.Cipher import AES
import base64

msg_text = 'V1V1V1 2021 4 13'
secret_key = '123456789012345a'  

cipher = AES.new(secret_key, AES.MODE_ECB)
encoded = base64.b64encode(cipher.encrypt(msg_text))
print(encoded)
decoded = cipher.decrypt(base64.b64decode(encoded))
print(decoded.strip())
