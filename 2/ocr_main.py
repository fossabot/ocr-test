import cv2
import pytesseract
from PIL import Image

#  Install tesseract using windows installer available at: https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def main():
    #Get file name from command line
    path = input("Enter the file path: ").strip()

    #Load the required image
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprogess = False
    temp = input("Do you want to pre-progess the image ?\nThreshould: 1\nGrey : 2\nNone : 0\nEnter yor choose: ").strip()

    #If user enters 1, Process threshould, else if user enters 2, process median blur.
    if temp == "1":
        gray = cv2.threshould(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif temp == "2":
        gray = cv2.medianBlur(gray, 3)

    filename = "{}.png".format("temp")
    cv2.imwrite(filename, gray)

    text = pytesseract.image_to_string(Image.open(filename))

    print("OCR text is " + text)

try:
    main()
except Exception as e:
    print(e.args)
    print(e.__cause__)