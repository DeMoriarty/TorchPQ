
def per(n):
  results=[]
  for i in range(1<<n):
    s=bin(i)[2:]
    s='0'*(n-len(s))+s
    a=''.join(list(map(str,list(s))))
    results.append(a)
  return results

print(per(2))
print(per(3))
print(per(4))

def decode(self, data):
  n=self.n
  obs = (
    [data[i:i+n] for i in range(0, len(data), n)]
    )
  start_metric = {i:0 for i in per(self.L)}
  state_machine = {
  }
    # current state, possible branches, branch information