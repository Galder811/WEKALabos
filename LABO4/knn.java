import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length<1){
            System.out.println("Ez daude arg gehiegi");
        }

        System.out.println("Paths: ");
        for (int i = 0; i< args.length;i++){
            System.out.println(args[i]);
        }

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        IBk knn = new IBk();
        LinearNNSearch chebyshevDistance = new LinearNNSearch();
        chebyshevDistance.setDistanceFunction(new ChebyshevDistance());
        LinearNNSearch euclideanDistance = new LinearNNSearch();
        euclideanDistance.setDistanceFunction(new EuclideanDistance());
        LinearNNSearch manhattanDistance = new LinearNNSearch();
        manhattanDistance.setDistanceFunction(new ManhattanDistance());
        LinearNNSearch filteredDistance = new LinearNNSearch();
        filteredDistance.setDistanceFunction(new FilteredDistance());
        LinearNNSearch minkowskiDistance = new LinearNNSearch();
        minkowskiDistance.setDistanceFunction(new MinkowskiDistance());
        LinearNNSearch[] distantziak = new LinearNNSearch[]{chebyshevDistance, euclideanDistance, manhattanDistance, filteredDistance, minkowskiDistance};
        SelectedTag[] tags = new SelectedTag[]{new SelectedTag(1, IBk.TAGS_WEIGHTING), new SelectedTag(2, IBk.TAGS_WEIGHTING), new SelectedTag(4, IBk.TAGS_WEIGHTING)};
        int kaux = 0;
        LinearNNSearch daux = null;
        SelectedTag waux = null;
        double fmeasureaux = 0.0;
        double fmeasuremax = 0.0;
        Evaluation eval = new Evaluation(data);

        for(int k = 1; k < data.numInstances() / 4; ++k) {
            knn.setKNN(k);
            LinearNNSearch[] var23 = distantziak;
            int var22 = distantziak.length;

            for(int var21 = 0; var21 < var22; ++var21) {
                LinearNNSearch d = var23[var21];
                knn.setNearestNeighbourSearchAlgorithm(d);
                SelectedTag[] var27 = tags;
                int var26 = tags.length;

                for(int var25 = 0; var25 < var26; ++var25) {
                    SelectedTag w = var27[var25];
                    knn.setDistanceWeighting(w);
                    eval.crossValidateModel(knn, data, 3, new Debug.Random(1));
                    fmeasureaux = eval.weightedFMeasure();
                    if (fmeasureaux > fmeasuremax) {
                        fmeasuremax = fmeasureaux;
                        kaux = k;
                        daux = d;
                        waux = w;
                    }
                }
            }
        }

        System.out.println(" Fmeasure optimoa: " + fmeasuremax + ", hurrengo parametroekin lortu dena: ");
        System.out.println(" k = " + kaux);
        System.out.println(" d = " + daux.distanceFunctionTipText());
        System.out.println(" w = " + waux);
        fitxategia_idatzi(eval, args[1]);
    }

    private static void fitxategia_idatzi(Evaluation ev, String direktorio) {
        try {
            FileWriter file = new FileWriter(direktorio);
            PrintWriter pw = new PrintWriter(file);
            pw.println("Direktorioa -->" + direktorio);
            pw.println("Nahasmen matrizea -->" + ev.toMatrixString());
            pw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
