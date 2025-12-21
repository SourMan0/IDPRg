import jax_unirep
import jax_unirep.layers as L

print("jax_unirep version:", getattr(jax_unirep, "__version__", "unknown"))
print([name for name in dir(L) if "mLSTM" in name])
