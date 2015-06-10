package org.djkazic.cortex.NLPMod.main;

import java.io.File;
import java.io.IOException;
import net.iharder.jpushbullet2.PushbulletClient;
import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.djkazic.cortex.NLPMod.console.LoggedPrintStream;

public class App {

	private static Word2Vec vector;
	private static boolean broadCast = true;
	private static long lastSent = 0;
	private static long waitTime = 1000;
	
	public static void main(String[] args) throws ResourceInitializationException, IOException {
		File path = new File("vmodel.dat");
		if(path.exists()) {
			System.out.println("VModel detected, loading...");
			vector = SerializationUtils.readObject(path);
			System.out.println(vector.wordsNearest("man", 4));
		} else {
			System.out.println("No existing VModel, training...");
			File training = new File("trainDir/");
			SentenceIterator iter = new FileSentenceIterator(new SentencePreProcessor() {
				private static final long serialVersionUID = -8826025951397717234L;

				public String preProcess(String sentence) {
					return new InputHomogenization(sentence).transform();
				}
			}, training);
			TokenizerFactory tf = new UimaTokenizerFactory();
			vector = new Word2Vec.Builder().windowSize(5).layerSize(300)
					.iterate(iter).tokenizerFactory(tf).build();
			(new Thread() {
				public void run() {
					LoggedPrintStream lpsOut = LoggedPrintStream.create(System.out);
					boolean notifiedBinary = false;
					while(App.broadCast) {
						try {
							String last = null;
							System.setOut(lpsOut);
							last = lpsOut.getBuf().toString();
							//System.setOut(lpsOut.underlying);
							if(last != null) {
								if(last.contains("Building binary tree") && !notifiedBinary) {
									pushNotification("Constructing binary tree structure");
									notifiedBinary = true;
								} else if(last.contains("Sent")) {
									//int sentIndex = last.indexOf("Sent");
									String[] split = last.split(" ");								
									int numberSent = Integer.parseInt(split[split.length - 1].trim());
									if(numberSent < 2000000 && (numberSent % 1000000) == 0
									   || numberSent > 2000000 && (numberSent % 200000) == 0) {
										pushNotification("Lines sent: " + numberSent);
										//System.out.println("num passed: " + numberSent);
									}
								}
							}
						} catch (Exception e) {
							e.printStackTrace();
						}
						try {
							Thread.sleep(400);
						} catch (InterruptedException e) {}
					}
				}
			}).start();
			vector.fit();
			pushNotification("Finished training");
			SerializationUtils.saveObject(vector, path);
		}
		//vector.getWordVectorMatrix("oil").
		//System.out.printf("%f\n", vector.similarity("oil", "gas"));
	}
	
	private static void pushNotification(String pushText) {
		try {
			if(System.currentTimeMillis() > (lastSent + waitTime)) {
				PushbulletClient client = new PushbulletClient("krOLMrMTBdeIIx8Qxn3uLY6Fh9Dhv436");
		        client.sendNote(null, "NLP Status", pushText);
		        lastSent = System.currentTimeMillis();
			}
		} catch (Exception e) {
			e.printStackTrace();
			waitTime = 1800; //Adds a half hour pause for rate-limiting
		}
	}
}
