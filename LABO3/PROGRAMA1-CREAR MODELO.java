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
import java.text.SimpleDateFormat;
import java.util.Date;

public class Main {

    public static void main(String[] args) throws Exception {
        // Verificar los argumentos de entrada
        if (args.length != 3) {
            System.err.println("Uso: java NaiveBayesModelCreation <data.arff> <NB.model> <QualityEstimation.txt>");
            System.exit(1);
        }
        // CARGAR LOS DATOS
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

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

        // CREAR EL .model
        weka.core.SerializationHelper.write(args[1], model);

        //////////////////////////////////////////////////////////////////////

        // CREAR LA EVALUACIÓN PARA APLICAR K-fold
        Evaluation evaluacion = entrenarNaiveBayesConValidacionCruzada(data, 5);

        //Obtener matrix
        String resultados = evaluacion.toMatrixString("Matriz de Confusión") + "\n";
        resultados += "Métricas de Precisión:\n" + evaluacion.toClassDetailsString();

        // CREAR ARCHIVO
        String rutaArchivoARFF = args[0];
        String rutaArchivoResultados = args[2];

        guardarResultadosEnArchivo(rutaArchivoARFF, rutaArchivoResultados, resultados);
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
    private static Evaluation entrenarNaiveBayesConValidacionCruzada(Instances data, int numFolds) throws Exception{
        final NaiveBayes model2 = new NaiveBayes();
        Evaluation evaluacion = new Evaluation(data);
        for(int i=1;i<=numFolds;i++){
            evaluacion.crossValidateModel(model2, data, numFolds, new java.util.Random(i));
        }
        return evaluacion;
    }

    private static String obtenerFechaActual() {
        SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date date = new Date();
        return formatter.format(date);
    }

}
