import numpy as np
from recsim.simulator.environment import SingleUserEnvironment

from environment.netflix.simulator.document import StaticCandidateSet


class Clock(object):

    def __init__(self):
        self.date = None

    def get_current_date(self):
        return self.date

    def set_date(self, date):
        self.date = date


class NetflixEnvironment(SingleUserEnvironment):

    def __init__(self,
                 user_model,
                 document_sampler,
                 num_candidates,
                 slate_size,
                 resample_documents=True):
        self._candidate_set = StaticCandidateSet()
        super().__init__(user_model, document_sampler, num_candidates, slate_size, resample_documents)
        self._current_documents = self._candidate_set.create_observation()

    def _do_resample_documents(self):
        for _ in range(self._num_candidates):
            self._candidate_set.add_document(self._document_sampler.sample_document())

    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        if self._resample_documents:
            self._do_resample_documents()
            self._current_documents = self._candidate_set.create_observation()
        return user_obs, self._current_documents

    def reset_sampler(self):
        """Resets the relevant samplers of documents and user/users."""
        self._document_sampler.reset_sampler()
        self._user_model.reset_sampler()

    def step(self, slate):
        """Executes the action, returns next state observation and reward.

        Args:
          slate: An integer array of size slate_size, where each element is an index
            into the set of current_documents presented

        Returns:
          user_obs: A gym observation representing the user's next state
          doc_obs: A list of observations of the documents
          responses: A list of AbstractResponse objects for each item in the slate
          done: A boolean indicating whether the episode has terminated
        """

        assert (len(slate) <= self._slate_size
                ), 'Received unexpectedly large slate size: expecting %s, got %s' % (
            self._slate_size, len(slate))

        # Get the documents associated with the slate
        mapped_slate = [self._candidate_set.doc_ids[x] for x in slate]
        documents = self._candidate_set.get_documents(mapped_slate)
        # Simulate the user's response
        responses = self._user_model.simulate_response(documents)

        # Update the user's state.
        self._user_model.update_state(documents, responses)

        # Update the documents' state.
        self._document_sampler.update_state(documents, responses)

        # Obtain next user state observation.
        user_obs = self._user_model.create_observation()

        # Check if reaches a terminal state and return.
        done = self._user_model.is_terminal()

        # Optionally, recreate the candidate set to simulate candidate
        # generators for the next query.
        if self._resample_documents:
            self._do_resample_documents()
            self._current_documents = self._candidate_set.create_observation()

        return user_obs, self._current_documents, responses, done


def ratings_reward(responses):
    return np.sum([response.rating for response in responses])

