import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class Main {

    public static void main(String[] args) throws Exception {
        // Verificar los argumentos de entrada
        if (args.length < 5) {
            System.err.println("Uso: java NaiveBayesModelCreation <data.arff> <NB.model> <QualityEstimation.txt>");
            System.exit(1);
        }
        // CARGAR LOS DATOS
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        //hacer hold out con los datos
        aplicarHoldOut(data,args[2]);
        //kfold
        entrenarNaiveBayesConValidacionCruzada(data, 5,args[2]);
        //////////////////////////////////////////////////////////////////////

        // CREAR LA EVALUACIÓN PARA APLICAR K-fold
        //Evaluation evaluacion = entrenarNaiveBayesConValidacionCruzada(data, 5);

        //Obtener matrix
        //String resultados = evaluacion.toMatrixString("Matriz de Confusión") + "\n";
        //resultados += "Métricas de Precisión:\n" + evaluacion.toClassDetailsString();

        // CREAR ARCHIVO
        //String rutaArchivoARFF = args[0];
        //String rutaArchivoResultados = args[2];

        //guardarResultadosEnArchivo(rutaArchivoARFF, rutaArchivoResultados, resultados);

        // CREAR EL .model

        final NaiveBayes model = new NaiveBayes();
        model.buildClassifier(data);
        weka.core.SerializationHelper.write(args[1], model);

        bigarrenPrograma(model, args[3], args[4]);
        }

    private static void bigarrenPrograma(NaiveBayes model, String path1, String path2) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path1);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        BufferedWriter writer = new BufferedWriter(new FileWriter(path2));

        // Hacer predicciones para cada instancia en el conjunto de prueba
        writer.write("Instancia, Clase Predicha\n");
        for (int i = 0; i < data.numInstances(); i++) {
            double pred = model.classifyInstance(data.instance(i));
            writer.write((i+1) + ", " + data.classAttribute().value((int) pred) + "\n");
        }

        writer.close();
    }

    private static void aplicarHoldOut(Instances data, String path) throws Exception {
        // APLICAR HOLD OUT AL 70% PARA APLICAR EL FILTRO
        final Randomize filterRandom = new Randomize();
        filterRandom.setRandomSeed(1);
        filterRandom.setInputFormat(data);
        final Instances RandomData = Filter.useFilter(data, (Filter) filterRandom);
        System.out.println("Data total tiene estas instancias " + data.numInstances());

        // APLICAR EL FILTRO AL TRAIN
        final RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setPercentage(30.0);
        filterRemove.setInvertSelection(false);
        filterRemove.setInputFormat(RandomData);
        final Instances train = Filter.useFilter(RandomData, (Filter) filterRemove);
        System.out.println("Train tiene estas instancias " + train.numInstances());

        // APLICAR EL FILTRO AL DEV
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(RandomData);
        final Instances dev = Filter.useFilter(RandomData, (Filter) filterRemove);
        System.out.println("Dev tiene estas instancias " + dev.numInstances());

        // SET LAS CLASES
        train.setClassIndex(train.numAttributes() - 1);
        dev.setClassIndex(dev.numAttributes() - 1);

        // APLICAR NAIVE BAYES
        final NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);

        final Evaluation eval = new Evaluation(train);
        eval.evaluateModel((Classifier)model, dev, new Object[0]);
        System.out.println(eval.toMatrixString());
        fitxategiaSortu(eval, path,false);
    }

    private static void entrenarNaiveBayesConValidacionCruzada(Instances data, int numFolds,String path) throws Exception{
        final NaiveBayes model2 = new NaiveBayes();
        Evaluation evaluacion = new Evaluation(data);
        for(int i=1;i<=numFolds;i++){
            evaluacion.crossValidateModel(model2, data, numFolds, new java.util.Random(i));
        }
        fitxategiaSortu(evaluacion,path,true);
        //return evaluacion;
    }
    private static void guardarResultadosEnArchivo(String rutaArchivoARFF, String rutaArchivoResultados, String resultados) throws Exception {
        // Crear el archivo de resultados
        BufferedWriter writer = new BufferedWriter(new FileWriter(rutaArchivoResultados));
        // Escribir la fecha de ejecución y los argumentos
        writer.write("Fecha de ejecución: " + obtenerFechaActual() + "\n");
        writer.write("Argumentos de ejecución:\n");
        writer.write("Archivo ARFF: " + rutaArchivoARFF + "\n");
        writer.write("Archivo de Resultados: " + rutaArchivoResultados + "\n\n");
        // Escribir los resultados de la evaluación
        writer.write(resultados);
        writer.close();
    }


    private static String obtenerFechaActual() {
        SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date date = new Date();
        return formatter.format(date);
    }
    private static void fitxategiaSortu(final Evaluation eval, final String directory,Boolean borrar) {
        try {
            double acc=eval.pctCorrect();
            double inc=eval.pctIncorrect();
            double kappa=eval.kappa();
            double mae=eval.meanAbsoluteError();
            double rmse=eval.rootMeanSquaredError();
            double rae=eval.relativeAbsoluteError();
            double rrse=eval.rootRelativeSquaredError();
            final FileWriter file = new FileWriter(directory,borrar);
            final PrintWriter pw = new PrintWriter(file);
            pw.println("Fitxategia sortu:" + directory);
            final String data = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
            pw.println("Exekuzioa data--> " + data);
            pw.println("Nahasmen-Matrizea: " + eval.toMatrixString());
            pw.println("Métricas de Precisión:\n" + eval.toClassDetailsString());
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
