import requests

# Update this path to match the actual location of your image file
image_path = 'D:/anki/project/flowers/daisy/34665595995_13f76d5b60_n.jpg'

url = "http://127.0.0.1:5000/predict"
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

print(response.json())
