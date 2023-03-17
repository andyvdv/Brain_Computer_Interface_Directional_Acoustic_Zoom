from brian2hears import Filterbank
import numpy as np

class EnvelopeFromGammatoneFilterbank(Filterbank):
    """Converts the output of a GammatoneFilterbank to an envelope."""

    def __init__(self, source):
        """Initialize the envelope transformation.

        Parameters
        ----------
        source : Gammatone
            Gammatone filterbank output to convert to envelope
        """
        super().__init__(source)

        self.nchannels = 1

    def buffer_apply(self, input_):
        # 6. take absolute value of the input_
        input_abs = abs(input_)

        # 7. do power-law compression (example with exponent 0.6)
        input_compressed = pow(input_abs, 0.6)

        # 8. take the sum over the channels, to get a single envelope
        envelope = np.sum(input_compressed, axis=1)

        return envelope