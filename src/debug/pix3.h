/*==========================================================================;
 *
 *  Copyright (C) Microsoft Corporation.  All Rights Reserved.
 *
 *  File:       pix3.h
 *  Content:    PIX include file
 *              Minimal version for PIXLoadLatestWinPixGpuCapturerLibrary
 *
 ****************************************************************************/

#pragma once

#ifndef _PIX3_H_
#define _PIX3_H_

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

// PIX capture flags
#define PIX_CAPTURE_GPU (1 << 0)
#define PIX_CAPTURE_TIMING (1 << 1)

// PIX function typedefs
typedef HRESULT (WINAPI* BeginCaptureFunctionPtr)(DWORD CaptureFlags, const void* pParameters);
typedef HRESULT (WINAPI* EndCaptureFunctionPtr)(BOOL Discard);

// Load the latest WinPixGpuCapturer.dll from PIX installation
__forceinline HMODULE PIXLoadLatestWinPixGpuCapturerLibrary()
{
    // FIRST: Try to load from same directory as executable (simplest, most reliable)
    HMODULE dll = LoadLibraryW(L"WinPixGpuCapturer.dll");
    if (dll != NULL) {
        return dll;  // Success! DLL is in same folder as .exe
    }

    // SECOND: Check registry for PIX installation
    HKEY pixKey = NULL;
    LPCWSTR keyPath = L"Software\\Microsoft\\PIX";
    LSTATUS result = RegOpenKeyExW(HKEY_LOCAL_MACHINE, keyPath, 0, KEY_READ, &pixKey);

    if (result != ERROR_SUCCESS)
    {
        // Try HKEY_CURRENT_USER
        result = RegOpenKeyExW(HKEY_CURRENT_USER, keyPath, 0, KEY_READ, &pixKey);
    }

    if (result != ERROR_SUCCESS)
    {
        // No registry key - try hardcoded path first (faster)
        HMODULE dll = LoadLibraryW(L"C:\\Program Files\\Microsoft PIX\\2509.25\\WinPixGpuCapturer.dll");
        if (dll != NULL) {
            return dll;
        }

        // Fallback: search for latest version
        WCHAR pixPath[MAX_PATH];
        wcscpy_s(pixPath, MAX_PATH, L"C:\\Program Files\\Microsoft PIX\\");

        // Find latest version directory
        WIN32_FIND_DATAW findData;
        WCHAR searchPath[MAX_PATH];
        wcscpy_s(searchPath, MAX_PATH, pixPath);
        wcscat_s(searchPath, MAX_PATH, L"*.*");

        HANDLE hFind = FindFirstFileW(searchPath, &findData);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            WCHAR latestVersion[MAX_PATH] = L"";
            do
            {
                // Look for version directory (starts with digit)
                if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                    findData.cFileName[0] >= L'0' && findData.cFileName[0] <= L'9')
                {
                    if (wcscmp(findData.cFileName, latestVersion) > 0)
                    {
                        wcscpy_s(latestVersion, MAX_PATH, findData.cFileName);
                    }
                }
            } while (FindNextFileW(hFind, &findData));
            FindClose(hFind);

            if (latestVersion[0] != L'\0')
            {
                wcscat_s(pixPath, MAX_PATH, latestVersion);
                wcscat_s(pixPath, MAX_PATH, L"\\WinPixGpuCapturer.dll");

                return LoadLibraryW(pixPath);
            }
        }

        return NULL;
    }

    // Read path from registry
    WCHAR pixInstallPath[MAX_PATH];
    DWORD pathSize = sizeof(pixInstallPath);
    result = RegQueryValueExW(pixKey, L"InstallPath", NULL, NULL, (LPBYTE)pixInstallPath, &pathSize);
    RegCloseKey(pixKey);

    if (result != ERROR_SUCCESS)
    {
        return NULL;
    }

    // Append WinPixGpuCapturer.dll
    wcscat_s(pixInstallPath, MAX_PATH, L"\\WinPixGpuCapturer.dll");

    return LoadLibraryW(pixInstallPath);
}

// Get PIXBeginCapture function
__forceinline HRESULT PIXBeginCapture(DWORD CaptureFlags, const void* pParameters)
{
    static BeginCaptureFunctionPtr pixBeginCapture = NULL;

    if (pixBeginCapture == NULL)
    {
        // Get module that contains PIX functions
        HMODULE pixModule = GetModuleHandleW(L"WinPixGpuCapturer.dll");
        if (pixModule == NULL)
        {
            return E_FAIL;
        }

        pixBeginCapture = (BeginCaptureFunctionPtr)GetProcAddress(pixModule, "PIXBeginCapture");
        if (pixBeginCapture == NULL)
        {
            return E_FAIL;
        }
    }

    return pixBeginCapture(CaptureFlags, pParameters);
}

// Get PIXEndCapture function
__forceinline HRESULT PIXEndCapture(BOOL Discard)
{
    static EndCaptureFunctionPtr pixEndCapture = NULL;

    if (pixEndCapture == NULL)
    {
        // Get module that contains PIX functions
        HMODULE pixModule = GetModuleHandleW(L"WinPixGpuCapturer.dll");
        if (pixModule == NULL)
        {
            return E_FAIL;
        }

        pixEndCapture = (EndCaptureFunctionPtr)GetProcAddress(pixModule, "PIXEndCapture");
        if (pixEndCapture == NULL)
        {
            return E_FAIL;
        }
    }

    return pixEndCapture(Discard);
}

#ifdef __cplusplus
}
#endif

#endif // _PIX3_H_
