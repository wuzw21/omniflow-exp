"""The single MobileGPT contract used by AndroidWorld.

The only writable path is successful RunLog -> MobileGPT's own
Explore/Select/Derive authoring prompts -> official Memory.  Native-cold and
mechanical-direct bundles remain historical evidence outside the runtime
index; they cannot be selected or produced by the current pipeline.
"""

MOBILEGPT_MEMORY_MANIFEST = "mobilegpt_memory_manifest.json"
MOBILEGPT_MEMORY_SCHEMA = "omniflow.mobilegpt.semantic-memory.v1"
MOBILEGPT_SOURCE_METHOD = "mobilegpt_runlog_official_semantic_memory"
MOBILEGPT_PREP_TYPE = "mobilegpt_runlog_official_semantic_memory"
MOBILEGPT_LEARNING_MODE = "mobilegpt_runlog_official_semantic_conversion"
MOBILEGPT_AUDIT_SCHEMA = "omniflow.mobilegpt.audit.v2"
MOBILEGPT_EMBEDDING_MODEL = "GLM-Embedding-2"
MOBILEGPT_PHYSICAL_BACKEND = "mobilegpt_official_accessibility"

MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS = frozenset({MOBILEGPT_MEMORY_SCHEMA})
MOBILEGPT_SOURCE_METHOD_BY_SCHEMA = {
    MOBILEGPT_MEMORY_SCHEMA: MOBILEGPT_SOURCE_METHOD,
}
MOBILEGPT_PREP_TYPE_BY_SCHEMA = {
    MOBILEGPT_MEMORY_SCHEMA: MOBILEGPT_PREP_TYPE,
}
MOBILEGPT_LEARNING_MODE_BY_SCHEMA = {
    MOBILEGPT_MEMORY_SCHEMA: MOBILEGPT_LEARNING_MODE,
}
MOBILEGPT_AUDIT_SCHEMA_BY_SCHEMA = {
    MOBILEGPT_MEMORY_SCHEMA: MOBILEGPT_AUDIT_SCHEMA,
}
MOBILEGPT_SUPPORTED_SOURCE_METHODS = frozenset({MOBILEGPT_SOURCE_METHOD})
MOBILEGPT_SUPPORTED_PREP_TYPES = frozenset({MOBILEGPT_PREP_TYPE})
