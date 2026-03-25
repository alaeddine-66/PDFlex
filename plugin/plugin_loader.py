"""
PDFlex - Plugin Loader
Dynamic loading of external classes via environment variables.

Inspired by Meta Platforms (FAIR) design patterns.
Allows plugging any implementation without modifying the source code.
"""
from __future__ import annotations

import importlib.util
import os
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Generic, Optional, Type, TypeVar

from loguru import logger

T = TypeVar("T")


class PluginLoader(Generic[T]):
    """
    Generic plugin loader driven by environment variables.
    """

    def __init__(self, plugin_name: str) -> None:
        """
        Args:
            plugin_name: Plugin name in lowercase (e.g., "llm", "extractor").
        """
        self._plugin_name = plugin_name
        env_prefix = plugin_name.upper()

        self._module_path: str = os.getenv(f"{env_prefix}_PLUGIN_MODULE_PATH", "")
        self._class_name: str = os.getenv(f"{env_prefix}_PLUGIN_CLASS_NAME", "")
        self._module_name: str = f"pdflex_plugin_{plugin_name}"

    @property
    def is_configured(self) -> bool:
        """
        Returns True if required environment variables are set and non-empty.
        """
        return bool(self._module_path) and bool(self._class_name)

    def load(self) -> Type[T]:
        """
        Load and return the plugin class from the external module.

        Returns:
            The class itself (not an instance) — you are responsible for instantiation.

        Raises:
            EnvironmentError : if environment variables are not set.
            FileNotFoundError: if the module file cannot be found.
            ValueError       : if the module cannot be loaded.
            AttributeError   : if the class does not exist in the module.
        """
        if not self.is_configured:
            raise EnvironmentError(
                f"Plugin '{self._plugin_name}' is not configured. "
                f"Set {self._plugin_name.upper()}_PLUGIN_MODULE_PATH "
                f"and {self._plugin_name.upper()}_PLUGIN_CLASS_NAME in your .env file."
            )

        if not os.path.isfile(self._module_path):
            raise FileNotFoundError(
                f"Plugin module not found: '{self._module_path}'. "
                f"Check {self._plugin_name.upper()}_PLUGIN_MODULE_PATH."
            )

        spec: Optional[ModuleSpec] = spec_from_file_location(
            self._module_name, self._module_path
        )
        if spec is None:
            raise ValueError(
                f"Unable to read Python module: '{self._module_path}'."
            )

        plugin_module: ModuleType = module_from_spec(spec)
        loader: Optional[Loader] = spec.loader
        if loader is None:
            raise ValueError(
                f"No loader available for: '{self._module_path}'."
            )

        loader.exec_module(plugin_module)

        if not hasattr(plugin_module, self._class_name):
            available = [
                name for name in dir(plugin_module)
                if not name.startswith("_")
            ]
            raise AttributeError(
                f"Class '{self._class_name}' not found in '{self._module_path}'. "
                f"Available classes: {available}. "
                f"Check {self._plugin_name.upper()}_PLUGIN_CLASS_NAME."
            )

        klass: Type[T] = getattr(plugin_module, self._class_name)
        logger.info(
            f"[Plugin] '{self._plugin_name}' loaded: "
            f"{self._class_name} from {self._module_path}"
        )
        return klass

    def __repr__(self) -> str:
        status = "configured" if self.is_configured else "not configured"
        return (
            f"<PluginLoader[{self._plugin_name}] {status} | "
            f"path={self._module_path or 'N/A'} | "
            f"class={self._class_name or 'N/A'}>"
        )
