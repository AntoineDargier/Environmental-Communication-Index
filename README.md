# Environmental-Communication-Index
End of study project, CentraleSupélec x ElevenStrategy, 2022-2023
Antoine Dargier, Martin Lanchon, Martin Ponchon, Alexandre Pradeilles

### Goal
Detection of environmental subjects in French media: using Wav2Vec2.0 for Speech-to-text from radio and NLP for text classification (for newspaper and radio transcriptions)

### Language
```Python```

### Contents
1. Project Organization and Management
2. Automatic Speech Recognition
   2.1 Models
   2.2 Evaluation
   2.3 Transcription pipeline on radio
3. Text Topic Classification
   3.1 Models
   3.2 Multi-categories classification
   3.3 Evaluation
4. Our dashboard
5. Conclusion

### Libraries
* ```TensorFlow```
* ```librosa```
* ```PyTorch```
* ```transformers```
* ```pandas```
* ```scikit-learn```

### Conclusion
In this project, we were able to transcribe a few hundred radio broadcasts, retrieve millions
of press articles, and classify them by topic. This allowed us to make a small visualisation
platform to showcase these results and the evolution of environmental topics.

This subject was extremely formative for us because we learned about many state-of-the-art
models in an extensive literature search. We were able to develop our own method or use
pre-trained tools. We were trained on many of the key data science skills, from scraping,
call API, NLP, cloud and visualisation, which made this project extremely comprehensive.
We really enjoyed the opportunity to see many different tools and to do a project ” end-to-end”, from start to finish. Dealing with environmental issues that affect us was even more
motivating.

Today, we consider that we can trust the results given to the platform, the classification
results being good. However, an important limitation of the current tool is that we do not
know how precisely these topics are addressed. This can range from a simple comment on
the weather to a real concrete environmental topic, to green-washing... This is why we think
that this subject can be continued and pursued, to improve the models used if possible, and
above all to study the way in which these subjects are raised. In addition, we would like to
study more precisely the classification on radio, which has a somewhat different format, with
many topics covered in one programme. Are 5-minute classifications really the best solution?
With more time and resources, we would have continued the project in this direction.
