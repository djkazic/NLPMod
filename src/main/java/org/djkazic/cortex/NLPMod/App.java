package org.djkazic.cortex.NLPMod;

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
	private static boolean trainOnline = false;
	private static long lastSent = 0;
	private static long waitTime = 1000;
	private static File path;
	
	public static void main(String[] args) throws ResourceInitializationException, IOException {
		path = new File("vmodel.dat");
		if(path.exists()) {
			System.out.println("VModel detected, loading...");
			vector = SerializationUtils.readObject(path);
			
			System.out.println(vector.similarity("man", "woman"));
			System.out.println(vector.similarity("day", "night"));
			System.out.println(vector.similarity("day", "gauntlet"));
			System.out.println(vector.similarity("chair", "treaty"));
			
			//TODO:
			//Condition for training online
			for(String s : args) {
				if(s.equals("--trainOnline")) {
					trainOnline = true;
					break;
				}
			}
			if(trainOnline) {
				trainAndSave(vector);
			}
		} else {
			System.out.println("No existing VModel, training...");
			trainAndSave(null);
		}
	}
	
	private static void trainAndSave(Word2Vec provided) {
		File training = new File("trainDir/");
		SentenceIterator iter = new FileSentenceIterator(new SentencePreProcessor() {
			private static final long serialVersionUID = -8826025951397717234L;

			public String preProcess(String sentence) {
				return new InputHomogenization(sentence).transform();
			}
		}, training);
		TokenizerFactory tf;
		try {
			tf = new UimaTokenizerFactory();
			if(provided == null) {
				vector = new Word2Vec.Builder().windowSize(5).layerSize(300)
						.iterate(iter).tokenizerFactory(tf).build();
			} else {
				vector = provided;
				System.out.println("Online training vectors");
			}
			vector.fit();
		} catch (Exception e) { e.printStackTrace(); }
		startListener();
		pushNotification("Finished training");
		SerializationUtils.saveObject(vector, path);
		System.out.println("Network solidified");
	}
	
	private static void startListener() {
		(new Thread() {
			public void run() {
				LoggedPrintStream lpsOut = LoggedPrintStream.create(System.out);
				while(App.broadCast) {
					try {
						String last = null;
						System.setOut(lpsOut);
						last = lpsOut.getBuf().toString();
						if(last != null) {
							if(last.contains("Building binary tree")) {
								App.broadCast = false;
								pushNotification("Constructing binary tree structure");
							} else if(last.contains("Sent")) {
								//int sentIndex = last.indexOf("Sent");
								String[] split = last.split(" ");								
								int numberSent = -1;
								try {
									numberSent = Integer.parseInt(split[split.length - 1].trim());
								} catch (Exception e) {
									continue;
								}
								if(numberSent < 2000000 && (numberSent % 1000000) == 0
								   || numberSent > 2000000 && (numberSent % 200000) == 0) {
									pushNotification("Lines sent: " + numberSent);
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
	}
	
	private static void pushNotification(String pushText) {
		try {
			if(System.currentTimeMillis() > (lastSent + waitTime)) {
				PushbulletClient client = new PushbulletClient("krOLMrMTBdeIIx8Qxn3uLY6Fh9Dhv436");
		        client.sendNote(null, "NLP Status", pushText);
		        lastSent = System.currentTimeMillis() + 500; //Buffer of 500ms of wait time
			}
		} catch (Exception e) {
			e.printStackTrace();
			waitTime = 1800000; //Adds a half hour pause for rate-limiting
		}
	}
}