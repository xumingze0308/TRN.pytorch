import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=inplace),
    )

class GeneralizedTRN(nn.Module):
    def __init__(self, args):
        super(GeneralizedTRN, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.feature_extractor = build_feature_extractor(args)
        # TODO: Support more fusion methods
        if True:
            self.future_size = self.feature_extractor.fusion_size
            self.fusion_size = self.feature_extractor.fusion_size * 2

        self.hx_trans = fc_relu(self.hidden_size, self.hidden_size)
        self.cx_trans = fc_relu(self.hidden_size, self.hidden_size)
        self.fusion_linear = fc_relu(self.num_classes, self.hidden_size)
        self.future_linear = fc_relu(self.hidden_size, self.future_size)

        self.enc_drop = nn.Dropout(self.dropout)
        self.enc_cell = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.dec_drop = nn.Dropout(self.dropout)
        self.dec_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def encoder(self, camera_input, sensor_input, future_input, enc_hx, enc_cx):
        fusion_input = self.feature_extractor(camera_input, sensor_input)
        fusion_input = torch.cat((fusion_input, future_input), 1)
        enc_hx, enc_cx = \
                self.enc_cell(self.enc_drop(fusion_input), (enc_hx, enc_cx))
        enc_score = self.classifier(self.enc_drop(enc_hx))
        return enc_hx, enc_cx, enc_score

    def decoder(self, fusion_input, dec_hx, dec_cx):
        dec_hx, dec_cx = \
                self.dec_cell(self.dec_drop(fusion_input), (dec_hx, dec_cx))
        dec_score = self.classifier(self.dec_drop(dec_hx))
        return dec_hx, dec_cx, dec_score

    def step(self, camera_input, sensor_input, future_input, enc_hx, enc_cx):
        # Encoder -> time t
        enc_hx, enc_cx, enc_score = \
                self.encoder(camera_input, sensor_input, future_input, enc_hx, enc_cx)

        # Decoder -> time t + 1
        dec_score_stack = []
        dec_hx = self.hx_trans(enc_hx)
        dec_cx = self.cx_trans(enc_cx)
        fusion_input = camera_input.new_zeros((camera_input.shape[0], self.hidden_size))
        future_input = camera_input.new_zeros((camera_input.shape[0], self.future_size))
        for dec_step in range(self.dec_steps):
            dec_hx, dec_cx, dec_score = self.decoder(fusion_input, dec_hx, dec_cx)
            dec_score_stack.append(dec_score)
            fusion_input = self.fusion_linear(dec_score)
            future_input = future_input + self.future_linear(dec_hx)
        future_input = future_input / self.dec_steps

        return future_input, enc_hx, enc_cx, enc_score, dec_score_stack

    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        future_input = camera_inputs.new_zeros((batch_size, self.future_size))
        enc_score_stack = []
        dec_score_stack = []

        # Encoder -> time t
        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, enc_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step],
                future_input, enc_hx, enc_cx,
            )
            enc_score_stack.append(enc_score)

            # Decoder -> time t + 1
            dec_hx = self.hx_trans(enc_hx)
            dec_cx = self.cx_trans(enc_cx)
            fusion_input = camera_inputs.new_zeros((batch_size, self.hidden_size))
            future_input = camera_inputs.new_zeros((batch_size, self.future_size))
            for dec_step in range(self.dec_steps):
                dec_hx, dec_cx, dec_score = self.decoder(fusion_input, dec_hx, dec_cx)
                dec_score_stack.append(dec_score)
                fusion_input = self.fusion_linear(dec_score)
                future_input = future_input + self.future_linear(dec_hx)
            future_input = future_input / self.dec_steps

        enc_scores = torch.stack(enc_score_stack, dim=1).view(-1, self.num_classes)
        dec_scores = torch.stack(dec_score_stack, dim=1).view(-1, self.num_classes)
        return enc_scores, dec_scores
