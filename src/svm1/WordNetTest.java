/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package svm1;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 *
 * @author hp
 */
public class WordNetTest {

    public void testDictionary() throws IOException {

        // construct the URL to the Wordnet dictionary directory
        String wnhome = "C:\\Program Files (x86)\\WordNet\\2.1\\dict";
        System.out.println(wnhome);
        String path = wnhome + File.separator + "dict";
        URL url = new URL("file", null, wnhome);

        // construct the dictionary object and open it
        IDictionary dict = new Dictionary(url);
        dict.open();

        // look up first sense of the word "dog "
        IIndexWord idxWord = dict.getIndexWord("dog", POS.NOUN);
        IWordID wordID = idxWord.getWordIDs().get(0);
        IWord word = dict.getWord(wordID);
        System.out.println("Id = " + wordID);
        System.out.println(" Lemma = " + word.getLemma());
        System.out.println(" Gloss = " + word.getSynset().getGloss());
    }
}
