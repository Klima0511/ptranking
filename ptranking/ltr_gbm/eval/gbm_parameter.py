

from ptranking.ltr_adhoc.eval.parameter import ScoringFunctionParameter

class GBMScoringFunctionParameter(ScoringFunctionParameter):
    """  """
    def __init__(self, debug=False, sf_id=None, sf_json=None):
        super(GBMScoringFunctionParameter, self).__init__(debug=debug, sf_id=sf_id, sf_json=sf_json)

    def default_para_dict(self):
        return self.default_gbdt_para_dict()

    def default_gbdt_para_dict(self):
        """
        A default setting of the hyper-parameters of the stump scoring function.
        """
        # the main tuning parameters are integrated within GBDT-related objects
        self.sf_para_dict = dict(sf_id=self.sf_id)
        return self.sf_para_dict

    def grid_search(self):
        if self.sf_id == 'gbdt':
            if self.use_json:
                self.sf_para_dict = dict(sf_id=self.sf_id)
                yield self.sf_para_dict
            else:
                self.sf_para_dict = dict(sf_id=self.sf_id)
                yield self.sf_para_dict
        else:
            return super().grid_search()

    def to_para_string(self, log=False):
        return ''