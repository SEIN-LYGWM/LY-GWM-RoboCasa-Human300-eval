"""Passive feature/state/score alignment. Does not generate or select actions."""

class PredictionProbe:
    def __init__(self):
        self.pending={}

    def observe(self,contexts,features,state,scorer,mean,scale):
        import torch
        packed=torch.cat([features.mean(dim=1),state],dim=1)
        scores=scorer((packed-mean)/scale)
        assert torch.isfinite(scores).all()
        rows=[]
        for i,c in enumerate(contexts):
            key=(c['task_index'],c['episode_index'])
            current=dict(observed_reward_logit=float(scores[i].item()),
                         previous_forecast_check=None)
            old=self.pending.pop(key,None)
            if old is not None:
                assert c['request_index']==old['context']['request_index']+1
                assert c['env_step_count']==old['context']['env_step_count']+16
                def mse(a,b):return float((a-b).square().mean().item())
                check=dict(source_context=old['context'],target_context=dict(c),
                    selected_index=old['selected_index'],
                    predicted_reward_logit=old['score'],
                    observed_reward_logit=current['observed_reward_logit'],
                    feature_mse=mse(old['features'],features[i]),
                    pooled_feature_mse=mse(old['features'].mean(dim=0),features[i].mean(dim=0)),
                    persistence_feature_mse=mse(old['current_features'],features[i]),
                    state_mse=mse(old['state'],state[i]),
                    persistence_state_mse=mse(old['current_state'],state[i]))
                current['previous_forecast_check']=check
            else:
                assert c['request_index']==0,'Missing previous forecast'
            rows.append(current)
        return rows

    def remember(self,contexts,features,state,predictions,indices,logits):
        for i,(c,selected) in enumerate(zip(contexts,indices)):
            key=(c['task_index'],c['episode_index'])
            assert key not in self.pending
            self.pending[key]=dict(context=dict(c),selected_index=selected,score=logits[i][selected],
                features=predictions[selected]['features'][i].detach().clone(),
                state=predictions[selected]['state'][i].detach().clone(),
                current_features=features[i].detach().clone(),current_state=state[i].detach().clone())
