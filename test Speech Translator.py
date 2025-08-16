from googletrans import Translator

translator = Translator()
result = translator.translate("Hello, how are you?", dest='te')  # English → Telugu
print(result.text)

result2 = translator.translate("I love Python programming", dest='fr')  # English → French
print(result2.text)
