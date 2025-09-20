import nltk
from nltk.corpus import wordnet
from transformers import pipeline

nltk.download('wordnet')
nltk.download('omw-1.4')


text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"""

text2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"""


def synonym_replace(sentence):
    words = sentence.split()
    new_words = []
    for word in words:
        syns = wordnet.synsets(word)
        if syns:
            new_word = syns[0].lemmas()[0].name().replace("_"," ")
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)

text1_nltk = synonym_replace(text1)
text2_nltk = synonym_replace(text2)

paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def hf_paraphrase(text):
    result = paraphraser("paraphrase: " + text, max_length=512, do_sample=False)
    return result[0]['generated_text']

text1_paraphrase = hf_paraphrase(text1)
text2_paraphrase = hf_paraphrase(text2)

summarizer = pipeline("summarization")

def hf_summarize(text):
    result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']

text1_summary = hf_summarize(text1)
text2_summary = hf_summarize(text2)

print("\nPIPELINE 1: NLTK Synonym Replacement\n")
print("Text 1:\n", text1_nltk, "\n")
print("Text 2:\n", text2_nltk, "\n")

print("\nPIPELINE 2: Hugging Face T5 Paraphrasing\n")
print("Text 1:\n", text1_paraphrase, "\n")
print("Text 2:\n", text2_paraphrase, "\n")

print("\n===== PIPELINE 3: Hugging Face Summarization\n")
print("Text 1:\n", text1_summary, "\n")
print("Text 2:\n", text2_summary, "\n")
