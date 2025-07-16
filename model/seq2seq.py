import torch ## expression operrator
from model.encoder import EncoderRNN
from model.decoder import AttnDecoderRNN

class Seq2Seq:
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, max_length=20, device='cpu'):
        self.encoder = EncoderRNN(input_vocab_size, hidden_size).to(device)
        self.decoder = AttnDecoderRNN(hidden_size, output_vocab_size, max_length=max_length).to(device)
        self.device = device
        self.max_length = max_length

    def train_batch(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):
        encoder_hidden = self.encoder.init_hidden().to(self.device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei].to(self.device), encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[0]], device=self.device)  # SOS_token
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di].to(self.device))
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, target_tensor[di].to(self.device))
                if decoder_input.item() == 1:  # EOS_token
                    break
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item() / target_length

    def evaluate(self, input_tensor):
        with torch.no_grad():
            encoder_hidden = self.encoder.init_hidden().to(self.device)
            input_length = input_tensor.size(0)
            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei].to(self.device), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            decoder_input = torch.tensor([[0]], device=self.device)  # SOS_token
            decoder_hidden = encoder_hidden
            decoded_words = []
            for di in range(self.max_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                if topi.item() == 1:  # EOS_token
                    break
                else:
                    decoded_words.append(topi.item())
                decoder_input = topi.squeeze().detach()
            return decoded_words 
