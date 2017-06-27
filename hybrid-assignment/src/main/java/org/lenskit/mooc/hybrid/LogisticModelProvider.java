package org.lenskit.mooc.hybrid;

import com.google.common.base.StandardSystemProperty;
import org.lenskit.LenskitRecommender;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.inject.Transient;
import org.lenskit.util.ProgressLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.rmi.runtime.Log;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Trainer that builds logistic models.
 */
public class LogisticModelProvider implements Provider<LogisticModel> {
    private static final Logger logger = LoggerFactory.getLogger(LogisticModelProvider.class);
    private static final double LEARNING_RATE = 0.00005;
    private static final int ITERATION_COUNT = 100;

    private final LogisticTrainingSplit dataSplit;
    private final BiasModel baseline;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;
    private final int parameterCount;
    private final Random random;

    @Inject
    public LogisticModelProvider(@Transient LogisticTrainingSplit split,
                                 @Transient UserBiasModel bias,
                                 @Transient RecommenderList recs,
                                 @Transient RatingSummary rs,
                                 @Transient Random rng) {
        dataSplit = split;
        baseline = bias;
        recommenders = recs;
        ratingSummary = rs;
        parameterCount = 1 + recommenders.getRecommenderCount() + 1;
        random = rng;
    }

    @Override
    public LogisticModel get() {
        List<ItemScorer> scorers = recommenders.getItemScorers();
        double intercept = 0;
        double[] params = new double[parameterCount];

        LogisticModel current = LogisticModel.create(intercept, params);

        // TODO Implement model training

        for(int iter=0; iter<ITERATION_COUNT; iter++){
            List<Rating> data = dataSplit.getTuneRatings();
            Collections.shuffle(data, random);

            for(Rating rating: data){
                double bias = baseline.getIntercept() + baseline.getItemBias(rating.getItemId()) +
                                                        baseline.getUserBias(rating.getUserId());
                double y_ui = rating.getValue();

                double b1x1 = params[0]*bias;
                double b2x2 = params[1] * Math.log10(ratingSummary.getItemRatingCount(rating.getItemId()));
                double sigValues = intercept + b1x1 + b2x2;
                for (int i=0; i<recommenders.getItemScorers().size(); i++){
                    if(recommenders.getItemScorers().get(i).score(rating.getUserId(), rating.getItemId()) != null) {
                        sigValues += params[i + 2] * (recommenders.getItemScorers().get(i).score(rating.getUserId(), rating.getItemId()).getScore() - bias);
                    }
                }

                double sigm = LogisticModel.sigmoid(-y_ui * sigValues);

                intercept = LEARNING_RATE * y_ui * sigm;
                params[0] = LEARNING_RATE * y_ui * bias * sigm;
                params[1] = LEARNING_RATE * y_ui * Math.log10(ratingSummary.getItemRatingCount(rating.getItemId())) * sigm;
                for (int i=0; i<recommenders.getItemScorers().size(); i++){
                    if(recommenders.getItemScorers().get(i).score(rating.getUserId(), rating.getItemId()) != null) {
                        params[i + 2] = LEARNING_RATE * y_ui * sigm *
                                (recommenders.getItemScorers().get(i).score(rating.getUserId(), rating.getItemId()).getScore() - bias);
                    }
                }
            }
            for(int i=0; i<params.length; i++){
                System.out.print(params[i] + " ");
            }
            System.out.println("");
            current = LogisticModel.create(intercept, params);
        }

        return current;
    }

}
