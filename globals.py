import matplotlib.pyplot as plt
import scipy.signal as signal

### test data, For good inference you need 5 hours of music for each genre
### from each song the trainin get's train_chunks_per_file  chunks of duration_sec_train
### startgin from the begingin of the file     

#                                   / -number of files to download for this genere
#                    / - yt query  |  /- enable|disable [1|0] this download and training for this genre 
#                   |              | |
GENERES = {'all': 'xyzabcd-fgh-ijk|0|1',  # keep this always 0. for tetgin all the files
           'insAI':'instrumental metal and heavy metal AI music|100|0',
           'amon': 'amon amarth|40|0',
           'goodbye': 'goodbye to gravity|30|0',
           'infected': 'infected rain instrumetal|20|0',
           'accept': 'accept instrumental|20|0',
           'phoenix': 'phoenix romania|20|0'}



def gen_path(p):
    return f'./DNL/{p}/'

def gen_proc(p):
    return f'./DNL/{p}/proc'

def gen_bps(p):
    return f'./DNL/{p}/rate'

def vae_path(p):
    """Final VQ-VAE weights for a genre."""
    return f'M/{p}-vqvae.pt'

def vae_chk_path(p):
    """Resumable training checkpoint for a genre."""
    return f'M/{p}-vqvae-checkpoint.pt'

def vae_best_path(p):
    """Best-loss VQ-VAE weights for a genre."""
    return f'M/{p}-vqvae-best.pt'

def gen_model_path(p):
    """Final VQ-VAE weights for a genre."""
    return f'M/{p}-gen.pt'

def gen_chk_path(p):
    """Final VQ-VAE weights for a genre."""
    return f'M/{p}-gen-checkpoint.pt'
    
class Ploter:
    def __init__(self):
        self.epoch_losses = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], marker='o', linestyle='-', color='b', label='Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Live VAE Loss Plot')
        self.ax.grid(True)
        self.ax.legend()

    def add_point(self, epoch_loss):
        # Skip non-finite values so the plot never crashes
        if epoch_loss is None or not (epoch_loss == epoch_loss) or abs(epoch_loss) == float("inf"):
            print(f"[Ploter] Skipping non-finite loss: {epoch_loss}")
            return
        self.epoch_losses.append(float(epoch_loss))
        if not self.epoch_losses:
            return
        epochs_range = list(range(1, len(self.epoch_losses) + 1))
        self.line.set_data(epochs_range, self.epoch_losses)
        self.ax.set_xlim(1, max(epochs_range) + 1)
        finite = [v for v in self.epoch_losses if v == v and abs(v) != float("inf")]
        if not finite:
            return
        ymin = min(finite) * 0.9
        ymax = max(finite) * 1.1
        if ymin == ymax:
            ymax = ymin + 1e-6
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        self.ax.set_ylim(ymin, ymax)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def __del__(self):
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception:
            pass

