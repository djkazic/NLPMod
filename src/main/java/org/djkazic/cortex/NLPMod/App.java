package org.djkazic.cortex.NLPMod;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;

import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
public class App {

	private static File path;
	private static Word2Vec vector;
	private static PrintStream oldOut;
	private static NullPrintStream nullOut;
	
	public static void main(String[] args) throws ResourceInitializationException, IOException {
		oldOut = System.out;
		nullOut = new NullPrintStream();
		
		path = new File("vmodel.dat");
		
		File googleBin = new File("google.bin");
		if(googleBin.exists()) {
			print("Loading Google binary model");
			vector = (Word2Vec) WordVectorSerializer.loadGoogleModel(googleBin, true);
		} else {
			print("Fallback to vModel");
			if(path.exists()) {
				print("VModel detected, loading...");
				vector = WordVectorSerializer.loadFullModel(path.getAbsolutePath());

				boolean trainOnline = false;
				if(trainOnline) {
					//TODO: implement online training
					trainAndSave(vector);
				}
			} else {
				print("No existing VModel, training...");
				trainAndSave(null);
			}
		}
		
		//Test vector
		print("Vector for king: ");
		print(Arrays.toString(vector.getWordVector("king")));
		print("");
		
		print("Vector for war: ");
		print(Arrays.toString(vector.getWordVector("war")));
		print("");
	}
	
	private static void trainAndSave(Word2Vec provided) {
		File training = new File("trainDir/");
		if(training.isDirectory() && training.list().length > 0) {
			print("Training with " + training.list().length + " input files");
			SentenceIterator iter = new FileSentenceIterator(new SentencePreProcessor() {
				private static final long serialVersionUID = -8826025951397717234L;

				public String preProcess(String sentence) {
					return new InputHomogenization(sentence).transform();
				}
			}, training);
			
			try {
				if(provided == null) {
					print("Offline training enabled");
					InMemoryLookupCache cache = new InMemoryLookupCache();
					vector = new Word2Vec.Builder().windowSize(5).layerSize(300)
							 .iterate(iter).tokenizerFactory(new UimaTokenizerFactory())
							 .vocabCache(cache)
							 .build();
				} else {
					vector = provided;
					print("Online training vectors");
				}
				vector.fit();
			} catch (Exception e) { e.printStackTrace(); }
			print("Network solidified");
			
			try {
				WordVectorSerializer.writeFullModel(vector, path.getName());
				print("Network serialized");
			} catch(Exception ex) {
				ex.printStackTrace(oldOut);
			}
		} else {
			print("Empty training dir");
		}
	}
	
	private static void print(String str) {
		System.setOut(oldOut);
		System.out.println(str);
		System.setOut(nullOut);
	}
}