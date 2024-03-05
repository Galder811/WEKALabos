import java.io.*;
import java.sql.Array;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Calendar;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Summarizable;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {
    public static void main(String[] args) throws  Exception {
        //cargar datos
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        //randomiza
        Randomize randomize = new Randomize();
        randomize.setRandomSeed(1);
        randomize.setInputFormat(data);
        data= Filter.useFilter(data,randomize);
        //crear filtro para stratified hold out
        StratifiedRemoveFolds filter =new StratifiedRemoveFolds();
        filter.setNumFolds(10);

        Instances train = null;
        Instances test = null;
        for (int f = 1; f<filter.getNumFolds(); f++) {
            filter.setInputFormat(data);
            filter.setFold(f);
            filter.setInvertSelection(false);
            Instances fold = Filter.useFilter(data, filter);

            if (f > 1 && f <= 7) {//%70 para el train
                if (train == null) {
                    train = fold;
                } else {
                    for (int i = 0; i < fold.numInstances(); i++) {
                        train.add(fold.get(i));
                    }
                }
            } else {//el %30 para test
                if (test == null) {
                    test = fold;
                } else
                    for (int i = 0; i < fold.numInstances(); i++) {
                        test.add(fold.get(i));
                    }
            }
        }
        test.setClassIndex(test.numAttributes()-1);
        train.setClassIndex(test.numAttributes()-1);
        // pasar el test a blind test.
        //work in pro.

        //Programa 2
        System.out.println("Atributu kopurua train: " + (train.numAttributes()-1));
        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setInputFormat(train);
        train = Filter.useFilter(train,attributeSelection);
        System.out.println("Atributu kopurua berrria train: " + (train.numAttributes()-1));

        final NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);
        weka.core.SerializationHelper.write(args[1], model);

        //hirugarren programa
        if (!test.equalHeaders(train)){
            System.out.println("Ez daude atributu berdinak.");
            //test egokitu
            System.out.println(test.numAttributes()-1);
            

        }
        //iragarpenak egin
        BufferedWriter writer = new BufferedWriter(new FileWriter(args[2]));

        // Hacer predicciones para cada instancia en el conjunto de prueba
        writer.write("Instancia, Clase Predicha\n");
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = model.classifyInstance(test.instance(i));
            writer.write((i+1) + ", " + test.classAttribute().value((int) pred) + "\n");
        }

        writer.close();

    }
}
