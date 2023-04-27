# Text summarizer 
The idea of this code is to create a easy-to use summarizer for english language, and then later in the development, portuguese. 
In this moment, the summary for english texts works fine, but in the portuguese stage of development, the character limit of GoogleTranslator
library limits the work to 5000 characters, making it unable to use to long texts, which is the main target of this work.
Also, it is not understand characters as 'ç', '^' and '~', which are common in the portuguese language.

To summarize from english, just set the translate to False, and if is the case, to work with short texts in portuguese (max 5000 characters)
set it to True (note issues above).