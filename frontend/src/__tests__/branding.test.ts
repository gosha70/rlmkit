/**
 * Tests for lib/branding.ts — product identity constants and the one-shot
 * localStorage key migration from the pre-rename (RLMKit) keys.
 */

import { describe, test, expect, beforeEach } from "vitest";
import {
  PRODUCT_NAME,
  STORAGE_KEY_ACTIVE_SESSION,
  STORAGE_KEY_CHAT_PROVIDER_ORDER,
  STORAGE_KEY_FILES_DRAFT,
  STORAGE_KEY_SELECTED_CHAT_PROVIDERS,
  migrateLegacyStorageKeys,
  storageKeyFilesForSession,
} from "@/lib/branding";

class MemoryStorage implements Storage {
  private map = new Map<string, string>();
  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.has(key) ? (this.map.get(key) as string) : null;
  }
  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

describe("branding constants", () => {
  test("product name and canonical key shapes", () => {
    expect(PRODUCT_NAME).toBe("RLM Studio");
    expect(STORAGE_KEY_ACTIVE_SESSION).toBe("rlm_studio_active_session");
    expect(STORAGE_KEY_SELECTED_CHAT_PROVIDERS).toBe("rlm_studio_selected_chat_providers");
    expect(STORAGE_KEY_CHAT_PROVIDER_ORDER).toBe("rlm_studio_cp_order");
    expect(STORAGE_KEY_FILES_DRAFT).toBe("rlm_studio_files_draft");
    expect(storageKeyFilesForSession("abc")).toBe("rlm_studio_files_abc");
  });
});

describe("migrateLegacyStorageKeys", () => {
  let storage: MemoryStorage;
  beforeEach(() => {
    storage = new MemoryStorage();
  });

  test("moves every known legacy key to its canonical name and removes the old one", () => {
    storage.setItem("rlmkit_active_session", "s1");
    storage.setItem("rlmkit_selected_chat_providers", '["a"]');
    storage.setItem("rlmkit-cp-order", '["a","b"]');
    storage.setItem("rlmkit_files_draft", "[]");
    storage.setItem("rlmkit_files_s1", '[{"id":1}]');
    storage.setItem("unrelated", "keep");

    expect(migrateLegacyStorageKeys(storage)).toBe(5);

    expect(storage.getItem(STORAGE_KEY_ACTIVE_SESSION)).toBe("s1");
    expect(storage.getItem(STORAGE_KEY_SELECTED_CHAT_PROVIDERS)).toBe('["a"]');
    expect(storage.getItem(STORAGE_KEY_CHAT_PROVIDER_ORDER)).toBe('["a","b"]');
    expect(storage.getItem(STORAGE_KEY_FILES_DRAFT)).toBe("[]");
    expect(storage.getItem(storageKeyFilesForSession("s1"))).toBe('[{"id":1}]');
    for (const legacy of [
      "rlmkit_active_session",
      "rlmkit_selected_chat_providers",
      "rlmkit-cp-order",
      "rlmkit_files_draft",
      "rlmkit_files_s1",
    ]) {
      expect(storage.getItem(legacy)).toBeNull();
    }
    expect(storage.getItem("unrelated")).toBe("keep");
  });

  test("does not overwrite a canonical value that already exists", () => {
    storage.setItem(STORAGE_KEY_ACTIVE_SESSION, "new");
    storage.setItem("rlmkit_active_session", "old");
    expect(migrateLegacyStorageKeys(storage)).toBe(1);
    expect(storage.getItem(STORAGE_KEY_ACTIVE_SESSION)).toBe("new");
    expect(storage.getItem("rlmkit_active_session")).toBeNull();
  });

  test("is a no-op when nothing legacy is present and leaves unknown rlmkit keys alone", () => {
    storage.setItem(STORAGE_KEY_ACTIVE_SESSION, "s1");
    storage.setItem("rlmkit_mystery", "x");
    expect(migrateLegacyStorageKeys(storage)).toBe(0);
    expect(storage.getItem("rlmkit_mystery")).toBe("x");
    expect(migrateLegacyStorageKeys(storage)).toBe(0);
  });

  test("tolerates a missing storage object", () => {
    expect(migrateLegacyStorageKeys(undefined)).toBe(0);
  });
});
