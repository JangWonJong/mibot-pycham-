import torch
from transformers import PreTrainedTokenizerFast

class Chatbot():
    def __init__(self):
        self.Q_TKN = "<usr>"
        self.A_TKN = "<sys>"
        self.BOS = '</s>'
        self.EOS = '</s>'
        self.MASK = '<unused0>'
        self.SENT = '<unused1>'
        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=self.BOS, eos_token=self.EOS, unk_token=self.UNK,  pad_token=self.PAD, mask_token=self.MASK)
    #모델 불러오기
    def execute_model(self):
        device = torch.device('cpu')
        PATH = 'save/chatbot_v50.pt'
        model = torch.load(PATH, map_location=device)
        model.eval()

        with torch.no_grad():
            while 1:
                q = input("user > ").strip()
                if q == "quit":
                    break
                a = ""
                while 1:
                    input_ids = torch.LongTensor(self.tokenizer.encode(self.Q_TKN + q + self.SENT+ self.A_TKN + a)).unsqueeze(dim=0)
                    pred = model(input_ids)
                    pred = pred.logits
                    gen = self.tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == self.EOS:
                        break
                    a += gen.replace("▁", " ")
                print("mibot > {}".format(a.strip()))

if __name__ == '__main__':
    Chatbot().execute_model()