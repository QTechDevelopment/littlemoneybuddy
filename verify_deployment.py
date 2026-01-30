#!/usr/bin/env python3
"""
Verification script for Vercel deployment
"""
import os
import sys

def check_file_exists(filepath):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    return exists

def main():
    print("🔍 Vercel Deployment Verification\n")
    print("=" * 50)
    
    required_files = [
        'vercel.json',
        '.vercelignore',
        'runtime.txt',
        'api/index.py',
        'requirements.txt',
        'DEPLOYMENT.md',
    ]
    
    print("\n📁 Required Files:")
    all_exist = all(check_file_exists(f) for f in required_files)
    
    print("\n📦 Dependencies:")
    try:
        with open('requirements.txt', 'r') as f:
            deps = f.read()
            has_starlette = 'starlette' in deps
            print(f"{'✅' if has_starlette else '❌'} starlette in requirements.txt")
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        all_exist = False
    
    print("\n🔧 Configuration Files:")
    try:
        import json
        with open('vercel.json', 'r') as f:
            config = json.load(f)
            has_builds = 'builds' in config
            has_routes = 'routes' in config
            print(f"{'✅' if has_builds else '❌'} builds configuration")
            print(f"{'✅' if has_routes else '❌'} routes configuration")
    except Exception as e:
        print(f"❌ Error reading vercel.json: {e}")
        all_exist = False
    
    print("\n🐍 Python Module:")
    try:
        sys.path.insert(0, os.getcwd())
        from api.index import app
        print(f"✅ API module imports successfully")
        print(f"✅ App type: {type(app).__name__}")
    except Exception as e:
        print(f"❌ Error importing API module: {e}")
        all_exist = False
    
    print("\n" + "=" * 50)
    if all_exist:
        print("✅ All deployment files are ready!")
        print("\n📝 Next steps:")
        print("1. Push to GitHub (already done)")
        print("2. Connect repository to Vercel")
        print("3. Deploy!")
        print("\n🌐 Alternative deployments: See DEPLOYMENT.md")
    else:
        print("❌ Some files are missing or have errors")
        print("Please check the errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()
