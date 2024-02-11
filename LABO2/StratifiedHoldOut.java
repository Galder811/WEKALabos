import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import javax.crypto.spec.PSource;

public class Main {
    public static void main(String[] args) throws Exception {

        // Menos de 3 NO
        if (args.length < 3) {
            System.out.println("Se requieren tres argumentos:\n1. Ruta del archivo train.arff\n2. Ruta del archivo dev.arff\n3. Ruta para guardar la evaluación");
            return;
        }

        // Printea args
        System.out.println("Argumentos:");
        for (String arg : args) {
            System.out.println(arg);
        }

        // Cargar datos
        ConverterUtils.DataSource Source = new ConverterUtils.DataSource(args[0]);
        Instances Data = Source.getDataSet();
        System.out.println(Data.numInstances());
        Data.setClassIndex(Data.numAttributes() - 1);

        //RANDOMIZE
        final Randomize filterRandom = new Randomize();
        filterRandom.setRandomSeed(1);
        filterRandom.setInputFormat(Data);
        final Instances RandomData = Filter.useFilter(Data, (Filter) filterRandom);
        System.out.println("Data tiene estas instancias: " + Data.numInstances());

        // StratifiedHoldOut -> Inicializar
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
        filter.setNumFolds(5);
        filter.setFold(1);
        filter.setInputFormat(RandomData);

        // Aplicar el filtro a Dev y train
        Instances Dev = Filter.useFilter(RandomData, filter);
        System.out.println("Dev: " + Dev.numInstances());
        filter.setInvertSelection(true);
        filter.setInputFormat(RandomData);
        Instances train = Filter.useFilter(RandomData, filter);
        System.out.println("Train: " + train.numInstances());


        // Obtener conjuntos de train y dev y guardaros en archivos .arff
        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(args[1]));
        saver.setInstances(train);
        saver.writeBatch();

        saver.setFile(new File(args[2]));
        saver.setInstances(Dev);
        saver.writeBatch();

        // Bayes
        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);

        // Evaluation
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, Dev);

        // Nahasmen matrix
        System.out.println(eval.toMatrixString());

        // Ejecutar creación de doc
        fitxategiaSortu(eval, args[3]);
    }

    private static void fitxategiaSortu(final Evaluation eval, final String directory) {
        try {
            double acc = eval.pctCorrect();
            double inc = eval.pctIncorrect();
            double kappa = eval.kappa();
            double mae = eval.meanAbsoluteError();
            double rmse = eval.rootMeanSquaredError();
            double rae = eval.relativeAbsoluteError();
            double rrse = eval.rootRelativeSquaredError();

            // Escribir resultados en el archivo de salida
            final FileWriter file = new FileWriter(directory);
            final PrintWriter pw = new PrintWriter(file);
            pw.println("Fitxategia sortu:" + directory);
            final String data = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
            pw.println("Exekuzioa data--> " + data);
            pw.println("Nahasmen-Matrizea: " + eval.toMatrixString());
            pw.println("\n" + "Correctly Classified Instances  " + acc + "\n");
            pw.println("Incorrectly Classified Instances  " + inc + "\n");
            pw.println("Kappa statistic  " + kappa + "\n");
            pw.println("Mean absolute error  " + mae + "\n");
            pw.println("Root mean squared error  " + rmse + "\n");
            pw.println("Relative absolute error  " + rae + "\n");
            pw.println("Root relative squared error  " + rrse + "\n");
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
