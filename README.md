# Melody_extraction
Code and Jupyter notebooks for melody extraction from polyphonic audio. UGP - 1.

*The report containing all the theoretical details of the work can be found in final_report folder*


*The code used is in the src folder*


#### Abstract

**Melody**: A popular definition is that ”the melody is the single (monophonic) pitch sequence that a
listener might reproduce if asked to whistle or hum a piece of polyphonic music, and that a listener
would recognize as being the essence of that music when heard in comparison.”

This definition is open to interpretation, and is a very subjective one. Different listeners might hum
different parts.


In a **vocal centric audio**, most will hum the vocal frequencies, but in instrumental pieces, different
people may follow different instruments as the melody.

In practice, research has focused on single source predominant fundamental frequency estimation,
that is melody is constrained to belong to a single sound source throughout the piece being analyzed,
where this sound source is considered to be the most predominant instrument or voice in the mixture.
Melody Extraction: Given a musical audio, output a frequency value for every time instant
representing the pitch of the dominant melodic line in the audio.

Melody Extraction from a polyphonic audio is a difficult task, in the sense that the term
melody is subjective, and cannot be given a generalised mathematical definition for all music. But
most of the methods till a few years back try to do so. A possible way to solve this problem is to
use data driven methods. Not many people have tried out this approach. Most recent ones are **Deep
Salience** by *Rachel M. Bittner et. all. in 2016* and **Source Filter NMF with CRNN** in *2017 by
Dogac Basaran et. all*. These methods are able to match the existing state of the art results.
The first one is heavily data dependent. The second tries solve this problem by bringing in a mid-
level representation of audio using **Source Filter - Non Negative Matrix Factorisation (SF-NMF)**,
instead of simply feeding the audio to the neural network.


*I have tried to solve this problem through an intelligent mid-level representation of audio,
improving upon SF-NMF, which also gives more control over the melody being extracted, solving
the subjectivity problem to some extent.*

