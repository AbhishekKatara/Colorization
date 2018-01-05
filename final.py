import Algorithmia

input = {
  "image": "/home/abhi/Pictures/a.png"
}
client = Algorithmia.client('simlqcj5reHWpRnf38uAC3ctBLv1')
algo = client.algo('deeplearning/ColorfulImageColorization/1.1.7')
print(algo.pipe(input))