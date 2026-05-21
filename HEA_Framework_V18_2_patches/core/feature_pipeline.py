# ============================================================
# FILE: core/feature_pipeline.py
# PATCHED — V18.2
# sklearn-compatible pipeline with safe fit/transform wiring
# ============================================================

from sklearn.base import BaseEstimator, TransformerMixin

from schema.feature_store import FeatureStore

from core.geometric_features import GeometricFeatureBuilder
from core.physical_descriptors import PhysicalDescriptorBuilder
from core.thermodynamic_descriptors import ThermodynamicDescriptorBuilder
from core.processing_features import ProcessingFeatureBuilder
from core.interaction_features import PhysicsInteractionGeneratorV2
from core.latent_features import LatentFeatureBuilder


class HEAFeaturePipelineV18_2(BaseEstimator, TransformerMixin):
    """
    Scientific feature pipeline.

    X must be dictionary-like:
        X["composition"] -> composition DataFrame
        X["processing"]  -> processing DataFrame
    """

    def __init__(
        self,
        element_data,
        thermocalc_engine,
        binary_mixing_enthalpies,
        latent_method="umap",
        latent_dimensions=8,
        enable_pairwise_interactions=False,
    ):
        self.element_data = element_data
        self.thermocalc_engine = thermocalc_engine
        self.binary_mixing_enthalpies = binary_mixing_enthalpies
        self.latent_method = latent_method
        self.latent_dimensions = latent_dimensions
        self.enable_pairwise_interactions = enable_pairwise_interactions

        self.geo = GeometricFeatureBuilder()
        self.phys = PhysicalDescriptorBuilder(element_data, binary_mixing_enthalpies)
        self.thermo = ThermodynamicDescriptorBuilder(thermocalc_engine)
        self.proc = ProcessingFeatureBuilder()
        self.inter = PhysicsInteractionGeneratorV2(enable_pairwise=enable_pairwise_interactions)
        self.latent = LatentFeatureBuilder(method=latent_method, n_components=latent_dimensions)

        self.feature_order_ = None
        self.block_columns_ = None
        self.is_fitted_ = False

    def _extract(self, X):
        return X["composition"], X["processing"]

    def _join_base(self, geo, phys, thermo, proc):
        return geo.join(phys).join(thermo).join(proc)

    def fit(self, X, y=None):
        X_comp, X_processing = self._extract(X)

        geo = self.geo.fit_transform(X_comp)
        phys = self.phys.transform(X_comp)
        thermo = self.thermo.fit_transform(X_comp)
        proc = self.proc.fit_transform(X_processing)

        base = self._join_base(geo, phys, thermo, proc)
        inter = self.inter.fit_transform(base)

        latent_input = base.join(inter)
        latent = self.latent.fit_transform(latent_input)

        store = FeatureStore()
        store.add_block("geometric", geo)
        store.add_block("physical", phys)
        store.add_block("thermodynamic", thermo)
        store.add_block("processing", proc)
        store.add_block("interaction", inter)
        store.add_block("latent", latent)

        full = store.concatenate()
        self.feature_order_ = full.columns.tolist()

        self.block_columns_ = {
            "geometric": geo.columns.tolist(),
            "physical": phys.columns.tolist(),
            "thermodynamic": thermo.columns.tolist(),
            "processing": proc.columns.tolist(),
            "interaction": inter.columns.tolist(),
            "latent": latent.columns.tolist(),
        }

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("HEAFeaturePipelineV18_2 must be fitted before transform().")

        X_comp, X_processing = self._extract(X)

        geo = self.geo.transform(X_comp)
        phys = self.phys.transform(X_comp)
        thermo = self.thermo.transform(X_comp)
        proc = self.proc.transform(X_processing)

        base = self._join_base(geo, phys, thermo, proc)
        inter = self.inter.transform(base)

        latent_input = base.join(inter)
        latent = self.latent.transform(latent_input)

        store = FeatureStore()
        store.add_block("geometric", geo)
        store.add_block("physical", phys)
        store.add_block("thermodynamic", thermo)
        store.add_block("processing", proc)
        store.add_block("interaction", inter)
        store.add_block("latent", latent)

        full = store.concatenate()
        return full.reindex(columns=self.feature_order_, fill_value=0.0)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
