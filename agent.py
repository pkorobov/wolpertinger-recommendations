from recsim.agent import AbstractEpisodicRecommenderAgent


class StaticAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, action_space, recommended_doc_id):
        super(StaticAgent, self).__init__(action_space)
        self.recommended_doc_id = recommended_doc_id

    def step(self, reward, observation):
        return [self.recommended_doc_id]
