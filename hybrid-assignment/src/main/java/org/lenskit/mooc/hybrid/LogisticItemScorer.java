package org.lenskit.mooc.hybrid;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.results.Results;
import org.lenskit.util.collections.LongUtils;
import sun.rmi.runtime.Log;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Item scorer that does a logistic blend of a subsidiary item scorer and popularity.  It tries to predict
 * whether a user has rated a particular item.
 */
public class LogisticItemScorer extends AbstractItemScorer {
    private final LogisticModel logisticModel;
    private final BiasModel biasModel;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;

    @Inject
    public LogisticItemScorer(LogisticModel model, UserBiasModel bias, RecommenderList recs, RatingSummary rs) {
        logisticModel = model;
        biasModel = bias;
        recommenders = recs;
        ratingSummary = rs;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        // TODO Implement item scorer

        List<Result> results = new ArrayList<>();

        RealVector coef = logisticModel.getCoefficients();

        for (Long item: items){
            double sigValues = logisticModel.getIntercept();
            double bias = biasModel.getIntercept() + biasModel.getItemBias(item) + biasModel.getUserBias(user);
            sigValues += bias * coef.getEntry(0);
            sigValues += Math.log10(ratingSummary.getItemRatingCount(item)) * coef.getEntry(1);

            for (int i=0; i<recommenders.getItemScorers().size(); i++){
                if(recommenders.getItemScorers().get(i).score(user, item) != null) {
                    sigValues += coef.getEntry(i + 2) * recommenders.getItemScorers().get(i).score(user, item).getScore();
                }
            }

            double res = LogisticModel.sigmoid(sigValues);
            results.add(Results.create(item, res));
        }

        return Results.newResultMap(results);
    }
}
