cat rock.mf classical.mf >rock_classical.mf
bash genseq.sh >inp_rock_classical.txt
../marsyas-0.5.0/bin/bextract -sv -mfcc -fe rock_classical.mf -w rock_classical.arff
grep -v '^@' MARSYAS_EMPTYrock_classical.arff |grep -v '^%' >>inp_rock_classical.txt



