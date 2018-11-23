import  torch
from    torch.nn import functional as F
import  time













def main():
    a = torch.randn(3, 2)

    w, b = torch.ones(2), torch.zeros(2)
    running_mean, running_var = torch.zeros(2), torch.ones(2)

    for i in range(1000):
        a = a + 0.1

        x= F.batch_norm(a, running_mean, running_var, w, b, training=False)
        print('\n\na:', a, '\nx:', x, '\n', 'w:', w, '\nrunning mean:', running_mean)

        time.sleep(5)





if __name__ == '__main__':
    main()