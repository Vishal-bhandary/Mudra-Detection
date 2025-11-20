"""
Main launcher for Bharatanatyam Mudra Detection System
"""
import os
import sys
import argparse

def check_setup():
    """Check if required files and folders exist"""
    issues = []
    
    # Check dataset
    if not os.path.exists("Bharatanatyam-Mudra-Dataset-master/Images"):
        issues.append("❌ Dataset not found at: Bharatanatyam-Mudra-Dataset-master/Images")
    
    # Check if landmarks extracted
    if not os.path.exists("data/processed/landmarks/dataset.pkl"):
        issues.append("⚠️  Landmarks not extracted. Run: python scripts/1_extract_landmarks.py")
    
    # Check if model trained
    if not os.path.exists("models/mudra_model.h5"):
        issues.append("⚠️  Model not trained. Run: python scripts/2_train_model.py")
    
    return issues

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║   Bharatanatyam Mudra Detection System               ║
    ║   Real-time Hand Gesture Recognition                 ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    parser = argparse.ArgumentParser(
        description='Bharatanatyam Mudra Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --extract      # Extract landmarks from dataset
  python run.py --train        # Train the model
  python run.py --detect       # Run live detection
  python run.py --all          # Run complete pipeline
        """
    )
    
    parser.add_argument('--extract', action='store_true',
                       help='Extract landmarks from images')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--detect', action='store_true',
                       help='Run live detection')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (extract → train → detect)')
    parser.add_argument('--check', action='store_true',
                       help='Check setup and requirements')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check setup
    if args.check or (not any([args.extract, args.train, args.detect, args.all])):
        print("Checking setup...\n")
        issues = check_setup()
        
        if not issues:
            print("✅ All systems ready!")
            print("\nNext steps:")
            print("  python run.py --detect     # Start live detection")
        else:
            print("Setup Status:\n")
            for issue in issues:
                print(f"  {issue}")
            
            print("\n📖 Quick Start:")
            print("  1. python run.py --extract    # Process dataset")
            print("  2. python run.py --train      # Train model")
            print("  3. python run.py --detect     # Run detection")
            print("\n  Or: python run.py --all       # Run everything")
        return
    
    # Run complete pipeline
    if args.all:
        print("Running complete pipeline...\n")
        
        print("STEP 1/3: Extracting landmarks...")
        os.system("python scripts/1_extract_landmarks.py")
        
        print("\nSTEP 2/3: Training model...")
        os.system("python scripts/2_train_model.py")
        
        print("\nSTEP 3/3: Starting live detection...")
        os.system("python scripts/3_live_detection.py")
        
        return
    
    # Run individual steps
    if args.extract:
        print("Extracting landmarks from dataset...\n")
        os.system("python scripts/1_extract_landmarks.py")
    
    if args.train:
        print("Training model...\n")
        if not os.path.exists("data/processed/landmarks/dataset.pkl"):
            print("❌ Error: Landmarks not found. Run --extract first.")
            return
        os.system("python scripts/2_train_model.py")
    
    if args.detect:
        print("Starting live detection...\n")
        if not os.path.exists("models/mudra_model.h5"):
            print("❌ Error: Model not found. Run --train first.")
            return
        os.system("python scripts/3_live_detection.py")

if __name__ == "__main__":
    main()