import java.io.*;
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
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {
    public static void main(String[] args) throws  Exception {

        /////////////////////EZ BADAUDE ARGUMENTURIK///////////////////////////////////////
        if(args.length < 2){
            System.out.println("\nJava proiektu hau erabiltzeko bi argumentu jarri behar dira: \n");
            System.out.println("1- Datuak dauzkan .arff dokumentuaren path-a.");
            System.out.println("2- Lortutako emaitzak non gordeko diren zehazten duen path-a. \n");
            System.out.println("Sartu berriro argumentuak!");
            return; // amaitu programa
        }
        /////////////////////ARGUMENTUAK ONDO SARTZEN BADIRA///////////////////////////////////////
        System.out.println("Bigarren laborategia lehenengo zatia: \n");
        System.out.println("Erabilitako path-ak:");
        for (int i=0; i< args.length;i++){
            System.out.println(args[i]);
        }

        ////////////////////////////////////////CARGAR DATOS//////////////////////////////////////////
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //////////////////////////////////////////Hold-Out//////////////////////////////////////////
        final Randomize filterRandom = new Randomize();
        filterRandom.setRandomSeed(1);
        filterRandom.setInputFormat(data);
        final Instances RandomData = Filter.useFilter(data, (Filter)filterRandom);
        System.out.println("Data total tiene estas instancias " + data.numInstances());

        final RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setPercentage(34.0);
        filterRemove.setInvertSelection(false);
        filterRemove.setInputFormat(RandomData);
        final Instances train = Filter.useFilter(RandomData, (Filter)filterRemove);
        System.out.println("Train tiene estas instancias " + train.numInstances());


        //final RemovePercentage filterRemove2 = new RemovePercentage();
        //filterRemove.setPercentage(34.0);
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(RandomData);
        final Instances dev = Filter.useFilter(RandomData, (Filter)filterRemove);
        System.out.println("Dev tiene estas instancias " + dev.numInstances());

        train.setClassIndex(train.numAttributes() - 1);
        dev.setClassIndex(dev.numAttributes() - 1);

        final NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);

        final Evaluation eval = new Evaluation(train);
        eval.evaluateModel((Classifier)model, dev, new Object[0]);
        System.out.println(eval.toMatrixString());
        fitxategiaSortu(eval, args[1]);
    }


    private static void fitxategiaSortu(final Evaluation eval, final String directory) {
        try {
            double acc=eval.pctCorrect();
            double inc=eval.pctIncorrect();
            double kappa=eval.kappa();
            double mae=eval.meanAbsoluteError();
            double rmse=eval.rootMeanSquaredError();
            double rae=eval.relativeAbsoluteError();
            double rrse=eval.rootRelativeSquaredError();
            final FileWriter file = new FileWriter(directory);
            final PrintWriter pw = new PrintWriter(file);
            pw.println("Fitxategia sortu:" + directory);
            final String data = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
            pw.println("Exekuzioa data--> " + data);
            pw.println("Nahasmen-Matrizea: " + eval.toMatrixString());
            pw.println("\n"+"Correctly Classified Instances  " + acc +"\n");
            pw.println("Incorrectly Classified Instances  " + inc+"\n");
            pw.println("Kappa statistic  " + kappa+"\n");
            pw.println("Mean absolute error  " + mae+"\n");
            pw.println("Root mean squared error  " + rmse+"\n");
            pw.println("Relative absolute error  " + rae+"\n");
            pw.println("Root relative squared error  " + rrse+"\n");
            pw.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
