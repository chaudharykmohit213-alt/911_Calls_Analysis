from PIL import Image

wordmatrix= Image.open('word_matrix.png')
#wordmatrix.show()
mask = Image.open('mask.png')
#mask.show()
print(wordmatrix.size)
print(mask.size)
mask2=mask.resize((1015, 559))
print(mask2.size)
mask2.putalpha(100)
mask2.show()
wordmatrix.paste(im=mask2,box=(0,0),mask=mask2)
wordmatrix.show()


