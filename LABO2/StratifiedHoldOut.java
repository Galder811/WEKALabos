import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

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

        // Cargeison datue
        ConverterUtils.DataSource trainSource = new ConverterUtils.DataSource(args[0]);
        Instances trainData = trainSource.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        // StratifiedHoldOut -> Inicializar
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
        filter.setInputFormat(trainData);
        filter.setNumFolds(2);

        // Aplicar el filtro a trainData
        Instances trainDev = Filter.useFilter(trainData, filter);

        // Obtener conjuntos de entrenamiento y desarrollo
        int numInstances = trainDev.numInstances();
        int splitIndex = (int) (numInstances * 0.8); //  80% para entrenamiento

        Instances train = new Instances(trainDev, 0, splitIndex);
        Instances dev = new Instances(trainDev, splitIndex, numInstances - splitIndex);

        // BayesianoPorculero
        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);

        // Evaluation
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, dev);

        // Nahasmen matrix
        System.out.println(eval.toMatrixString());

        // Ejecutar creación de doc
        fitxategiaSortu(eval, args[2]);
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
